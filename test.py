import sys
import os
import glob
import cv2
import numpy as np
import logging
import faiss

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QGraphicsView, QGraphicsScene, QFileDialog,
    QMessageBox, QStatusBar, QGraphicsRectItem, QGraphicsPixmapItem,
    QListWidget, QListWidgetItem, QCompleter, QLabel
)
from PyQt6.QtGui import QPixmap, QPen, QPainter, QColor, QBrush, QCursor
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QStringListModel

# --- Setup Logging ---
logging.basicConfig(
    filename='annotation_tool.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AnnotationScene(QGraphicsScene):
    """A QGraphicsScene subclass that emits signals for mouse events."""
    mousePressed = pyqtSignal(QPointF)
    mouseMoved = pyqtSignal(QPointF)
    mouseReleased = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mousePressed.emit(event.scenePos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouseReleased.emit(event.scenePos())
        super().mouseReleaseEvent(event)

class AnnotationApp(QMainWindow):
    """The main application window for image annotation."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuickDraw (Faiss Edition)")
        self.setGeometry(100, 100, 1200, 800)

        # --- FEATURE DETECTION ---
        # Using a moderate number of features is a good balance for speed and accuracy.
        self.sift = cv2.SIFT_create(nfeatures=800)

        # --- NEW: FAISS INDEX SETUP ---
        self.descriptor_dimension = 128  # SIFT descriptors are 128-dimensional
        self.is_faiss_trained = False
        self.min_refs_for_training = 10 # Minimum annotations before auto-annotation is enabled
        
        # 1. The coarse quantizer (for clustering the vector space)
        quantizer = faiss.IndexFlatL2(self.descriptor_dimension)
        # 2. The main index. nlist is the number of "visual words"/clusters.
        nlist = 100
        self.faiss_index = faiss.IndexIVFFlat(quantizer, self.descriptor_dimension, nlist)
        # Faiss requires float32. This is a crucial step.
        self.faiss_index.set_direct_map_type(faiss.DirectMap.Array)
        
        # 3. We still need our original reference list to hold metadata and for rebuilding the index.
        self.references = []
        # --- END FAISS SETUP ---

        self.image_list = []
        self.current_index = -1
        self.current_image_path = None
        self.current_rect_item = None
        self.start_point = QPointF()
        self.pixmap_item = None
        
        self.annotation_mode = False
        self.crosshair_h = None
        self.crosshair_v = None

        self.init_ui()
        logging.info("Application initialized with SIFT/Faiss.")

    def init_ui(self):
        """Initializes the user interface components."""
        # ... (UI initialization is identical to the previous version) ...
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        overall_layout = QVBoxLayout(central_widget)

        top_controls_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Select Folder")
        self.folder_btn.clicked.connect(self.open_folder)
        top_controls_layout.addWidget(self.folder_btn)

        self.path_label = QLabel("No folder selected.")
        self.path_label.setStyleSheet("font-style: italic; color: grey;")
        top_controls_layout.addWidget(self.path_label, 1)

        class_label = QLabel("Class:")
        top_controls_layout.addWidget(class_label)
        self.class_combo = QComboBox()
        self.class_combo.setMinimumHeight(30)
        self.class_combo.setMinimumWidth(200)
        self.class_combo.setEditable(True)
        self.class_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.completer = QCompleter(self)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.class_combo.setCompleter(self.completer)
        top_controls_layout.addWidget(self.class_combo)
        overall_layout.addLayout(top_controls_layout)

        main_content_layout = QHBoxLayout()
        self.image_list_widget = QListWidget()
        self.image_list_widget.setMaximumWidth(250)
        self.image_list_widget.currentItemChanged.connect(self.on_list_item_select)
        main_content_layout.addWidget(self.image_list_widget)

        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        main_content_layout.addWidget(right_panel_widget, 1)

        self.scene = AnnotationScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setMouseTracking(True)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        right_panel_layout.addWidget(self.view)
        
        pen = QPen(QColor(255, 0, 0, 120), 0.5)
        self.crosshair_h = self.scene.addLine(0, 0, 0, 0, pen)
        self.crosshair_v = self.scene.addLine(0, 0, 0, 0, pen)
        self.crosshair_h.setZValue(1)
        self.crosshair_v.setZValue(1)
        self.toggle_crosshairs(False)

        bottom_controls_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear (Esc)")
        self.clear_btn.clicked.connect(self.clear_annotation)
        bottom_controls_layout.addWidget(self.clear_btn)
        self.auto_btn = QPushButton("Auto-Annotate")
        self.auto_btn.setEnabled(False)
        self.save_btn = QPushButton("Save")
        self.delete_btn = QPushButton("Delete Image")
        self.delete_btn.setStyleSheet("background-color: #FF6347; color: white;")
        self.delete_btn.clicked.connect(self.delete_current_image)
        
        bottom_controls_layout.addWidget(self.auto_btn)
        bottom_controls_layout.addWidget(self.save_btn)
        bottom_controls_layout.addWidget(self.delete_btn)
        right_panel_layout.addLayout(bottom_controls_layout)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous (A)")
        self.next_btn = QPushButton("Next (D)")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        right_panel_layout.addLayout(nav_layout)
        overall_layout.addLayout(main_content_layout)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.auto_btn.clicked.connect(self.auto_annotate)
        self.save_btn.clicked.connect(self.save_annotation)
        self.next_btn.clicked.connect(self.next_image)
        self.prev_btn.clicked.connect(self.prev_image)
        self.scene.mousePressed.connect(self.scene_mouse_pressed)
        self.scene.mouseMoved.connect(self.scene_mouse_moved)
        self.scene.mouseReleased.connect(self.scene_mouse_released)
        
    def open_folder(self):
        # ... (This method is identical) ...
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder: return
        logging.info(f"Opening folder: {folder}")
        self.path_label.setText(folder)
        self.path_label.setStyleSheet("")

        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_list = []
        for ext in image_extensions: self.image_list.extend(glob.glob(os.path.join(folder, ext)))
        self.image_list = sorted(self.image_list)

        if not self.image_list:
            QMessageBox.warning(self, "No Images", "No images found.")
            return

        classes_path = os.path.join(folder, 'classes.txt')
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            self.class_combo.clear()
            self.class_combo.addItems(classes)
            completer_model = QStringListModel(classes, self)
            self.completer.setModel(completer_model)
        else:
            QMessageBox.warning(self, "File Not Found", "classes.txt not found.")
            self.class_combo.clear()

        self.populate_image_list_panel()
        self._build_initial_references()
        if self.image_list:
            self.current_index = 0
            self.load_image(self.current_index)
            
    def _build_initial_references(self):
        """Load all existing annotations into the reference list and build the Faiss index."""
        self.references = []
        logging.info("Building initial reference set from existing annotations.")
        self.statusBar.showMessage("Loading existing annotations...")
        QApplication.processEvents()

        for img_path in self.image_list:
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if not os.path.exists(txt_path): continue
            try:
                with open(txt_path, 'r') as f: data = f.readline().split()
                if len(data) < 5: continue
                class_id = int(data[0])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                kp, des = self.sift.detectAndCompute(img, None)
                if des is not None and len(des) > 0:
                    h, w = img.shape[:2]
                    cx, cy, bw, bh = map(float, data[1:5])
                    x_min, y_min = (cx - bw / 2) * w, (cy - bh / 2) * h
                    x_max, y_max = (cx + bw / 2) * w, (cy + bh / 2) * h
                    ref = {'image_path': img_path, 'kp': kp, 'des': des, 
                           'bbox': [x_min, y_min, x_max, y_max], 'class_id': class_id}
                    self.references.append(ref)
            except Exception as e:
                logging.error(f"Error loading reference for {img_path}: {e}")
        
        self._rebuild_faiss_index()
        self.statusBar.showMessage(f"Loaded {len(self.references)} references. Faiss index is {'trained' if self.is_faiss_trained else 'not trained'}.", 5000)

    def _rebuild_faiss_index(self):
        """Completely rebuilds the Faiss index from the self.references list.
        Necessary after loading or deleting many items.
        """
        if len(self.references) < self.min_refs_for_training:
            self.is_faiss_trained = False
            self.update_auto_annotate_button_status()
            logging.info(f"Not enough references ({len(self.references)}) to train Faiss index.")
            return

        self.statusBar.showMessage("Building Faiss index...", 0)
        QApplication.processEvents()
        logging.info("Rebuilding Faiss index...")

        # Reset and retrain
        self.faiss_index.reset()
        all_descriptors = np.vstack([ref['des'] for ref in self.references if ref['des'] is not None])
        all_descriptors = np.float32(all_descriptors)
        
        self.faiss_index.train(all_descriptors)
        self.faiss_index.add(all_descriptors)
        
        self.is_faiss_trained = True
        self.update_auto_annotate_button_status()
        self.statusBar.showMessage(f"Faiss index built with {self.faiss_index.ntotal} features from {len(self.references)} images.", 5000)
        logging.info(f"Faiss index rebuilt. Total vectors: {self.faiss_index.ntotal}")

    def auto_annotate(self, silent=False):
        """
        Auto-annotates using the high-speed Faiss index + geometric verification.
        """
        if not self.is_faiss_trained:
            if not silent:
                QMessageBox.information(self, "Auto-Annotate", f"Not ready. Please provide at least {self.min_refs_for_training} manual annotations.")
            return

        self.statusBar.showMessage("Auto-annotating...", 0)
        QApplication.processEvents()

        current_img_bgr = cv2.imread(self.current_image_path)
        if current_img_bgr is None: return
        current_img_gray = cv2.cvtColor(current_img_bgr, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(current_img_gray, None)
        if des1 is None or len(des1) == 0:
            if not silent: QMessageBox.warning(self, "Auto-Annotate", "No SIFT features found in the current image.")
            self.statusBar.clearMessage()
            return
        
        des1_float32 = np.float32(des1)

        # --- STAGE 1: Fast Candidate Search with Faiss ---
        # For each descriptor, find the 2 best matches in the entire index for ratio testing
        self.faiss_index.nprobe = 10  # Search 10 clusters, a good speed/accuracy trade-off
        k_neighbors = 2
        distances, indices = self.faiss_index.search(des1_float32, k_neighbors)

        # Map flat indices back to their original reference image
        candidate_votes = {}
        good_matches_by_candidate = {}

        # This map helps us find which reference a descriptor belongs to
        desc_cumsum = np.cumsum([0] + [len(ref['des']) for ref in self.references])

        for i in range(len(des1)):
            # Lowe's Ratio Test
            if distances[i][0] < 0.75 * distances[i][1]:
                matched_desc_index = indices[i][0]
                
                # Find which reference image this descriptor belongs to
                ref_idx = np.searchsorted(desc_cumsum, matched_desc_index + 1) - 1
                ref = self.references[ref_idx]
                ref_path = ref['image_path']

                # Vote for this reference image
                candidate_votes[ref_path] = candidate_votes.get(ref_path, 0) + 1
                
                if ref_path not in good_matches_by_candidate:
                    good_matches_by_candidate[ref_path] = []
                
                # Store the keypoint pair for homography
                query_kp_idx = i
                train_kp_idx_in_ref = matched_desc_index - desc_cumsum[ref_idx]
                good_matches_by_candidate[ref_path].append( (query_kp_idx, train_kp_idx_in_ref) )

        if not candidate_votes:
            if not silent: QMessageBox.information(self, "Auto-Annotate", "No promising candidates found.")
            self.statusBar.clearMessage()
            return
        
        # --- STAGE 2: Geometric Verification on Top Candidates ---
        sorted_candidates = sorted(candidate_votes.items(), key=lambda item: item[1], reverse=True)
        top_n_candidates = 5
        
        best_match = {'ref': None, 'score': -1, 'M': None}
        MIN_INLIER_COUNT = 10

        for ref_path, vote_count in sorted_candidates[:top_n_candidates]:
            ref = next((r for r in self.references if r['image_path'] == ref_path), None)
            if ref is None: continue
            
            match_pairs = good_matches_by_candidate[ref_path]
            
            src_pts = np.float32([kp1[m[0]].pt for m in match_pairs]).reshape(-1, 1, 2)
            dst_pts = np.float32([ref['kp'][m[1]].pt for m in match_pairs]).reshape(-1, 1, 2)

            if len(src_pts) < MIN_INLIER_COUNT: continue

            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if M is not None:
                inlier_count = np.sum(mask)
                if inlier_count > best_match['score'] and inlier_count >= MIN_INLIER_COUNT:
                    best_match['score'] = inlier_count
                    best_match['ref'] = ref
                    best_match['M'] = M
        
        # --- Apply the best match ---
        if best_match['ref'] is None:
            if not silent:
                QMessageBox.information(self, "Auto-Annotate", f"Failed to find a geometrically consistent match. Best candidate had {best_match['score']} inliers.")
            self.statusBar.clearMessage()
            return

        ref = best_match['ref']
        M = best_match['M']
            
        ref_bbox = ref['bbox']
        ref_pts = np.float32([
            [ref_bbox[0], ref_bbox[1]], [ref_bbox[0], ref_bbox[3]],
            [ref_bbox[2], ref_bbox[3]], [ref_bbox[2], ref_bbox[1]]
        ]).reshape(-1, 1, 2)
        
        transformed_pts = cv2.perspectiveTransform(ref_pts, M)
        
        x_min, y_min = np.min(transformed_pts, axis=0)[0]
        x_max, y_max = np.max(transformed_pts, axis=0)[0]
        
        h, w = current_img_gray.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        new_rect = QRectF(QPointF(x_min, y_min), QPointF(x_max, y_max))
        self.draw_bbox(new_rect, Qt.GlobalColor.blue)
        if ref['class_id'] < self.class_combo.count():
            self.class_combo.setCurrentIndex(ref['class_id'])
        
        self.statusBar.showMessage(f"Auto-annotation successful with {best_match['score']} inliers.", 5000)
        logging.info(f"Auto-annotation successful for {self.current_image_path} with {best_match['score']} inliers.")

    def save_annotation(self, silent=False):
        if not self.current_rect_item or self.class_combo.currentIndex() < 0:
            if not silent: QMessageBox.warning(self, "Save Error", "Please draw a bounding box and select a class.")
            return

        rect = self.current_rect_item.rect()
        img_w, img_h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
        if img_w == 0 or img_h == 0: return

        cx = (rect.left() + rect.width() / 2) / img_w
        cy = (rect.top() + rect.height() / 2) / img_h
        w = rect.width() / img_w
        h = rect.height() / img_h
        class_id = self.class_combo.currentIndex()

        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        self.update_references_and_index()
        self.update_list_item_status(self.current_index)
        self.draw_bbox(self.current_rect_item.rect(), Qt.GlobalColor.red)
        if not silent:
            QMessageBox.information(self, "Saved", f"Annotation saved to {os.path.basename(txt_path)}")
        logging.info(f"Annotation saved for {self.current_image_path}")
        
    def update_references_and_index(self):
        """Updates the reference list and adds the new entry to the Faiss index."""
        # Check if this is a new annotation or an update
        is_new_annotation = self.current_image_path not in [ref['image_path'] for ref in self.references]
        # Remove old entry if it exists
        self.references = [ref for ref in self.references if ref['image_path'] != self.current_image_path]

        img_gray = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: return
        
        kp, des = self.sift.detectAndCompute(img_gray, None)
        if des is None or len(des) == 0: return

        rect = self.current_rect_item.rect()
        new_ref = {
            'image_path': self.current_image_path, 'kp': kp, 'des': des,
            'bbox': [rect.left(), rect.top(), rect.right(), rect.bottom()],
            'class_id': self.class_combo.currentIndex()
        }
        self.references.append(new_ref)

        if is_new_annotation:
            # If the index isn't ready, we just collect data.
            # If it is ready, we add the new data incrementally.
            if self.is_faiss_trained:
                des_float32 = np.float32(des)
                self.faiss_index.add(des_float32)
                self.statusBar.showMessage(f"Added {len(des)} features to index. Total: {self.faiss_index.ntotal}", 3000)
                logging.info(f"Incrementally added {len(des)} features to Faiss index.")
            else:
                # If we've just reached the threshold, trigger a full build.
                if len(self.references) >= self.min_refs_for_training:
                    self._rebuild_faiss_index()
        else:
            # If an existing annotation is updated, it's safer to just rebuild the whole index.
            self._rebuild_faiss_index()
        
        self.update_auto_annotate_button_status()

    def delete_current_image(self):
        if self.current_index == -1:
            QMessageBox.warning(self, "No Image", "There is no image selected to delete.")
            return

        img_path = self.current_image_path
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        
        reply = QMessageBox.question(self, 'Confirm Deletion', 
                                     f"Are you sure you want to permanently delete this image and its annotation?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(img_path)
                logging.info(f"Deleted image file: {img_path}")
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    logging.info(f"Deleted annotation file: {txt_path}")
            except OSError as e:
                QMessageBox.warning(self, "Delete Error", f"Error deleting files:\n{e}")
                logging.error(f"Error deleting files for {img_path}: {e}")
                return

            deleted_index = self.current_index
            self.image_list.pop(deleted_index)
            self.image_list_widget.takeItem(deleted_index)
            # Remove from references and rebuild index
            self.references = [ref for ref in self.references if ref['image_path'] != img_path]
            self._rebuild_faiss_index()
            
            if not self.image_list:
                self.load_image(-1)
            else:
                next_index = min(deleted_index, len(self.image_list) - 1)
                self.load_image(next_index)

    def update_auto_annotate_button_status(self):
        if self.is_faiss_trained:
            self.auto_btn.setEnabled(True)
            self.auto_btn.setText("Auto-Annotate (Ready)")
            logging.info("Auto-annotate feature enabled.")
        else:
            self.auto_btn.setEnabled(False)
            self.auto_btn.setText(f"Auto-Annotate ({len(self.references)}/{self.min_refs_for_training})")
            
    # --- The rest of the methods are mostly unchanged ---

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_W: self.toggle_annotation_mode()
        elif key == Qt.Key.Key_A: self.prev_image()
        elif key == Qt.Key.Key_D: self.next_image()
        elif key == Qt.Key.Key_Escape: self.clear_annotation()
        else: super().keyPressEvent(event)

    def populate_image_list_panel(self):
        self.image_list_widget.clear()
        for i, img_path in enumerate(self.image_list):
            item = QListWidgetItem(os.path.basename(img_path))
            self.image_list_widget.addItem(item)
            self.update_list_item_status(i)

    def update_list_item_status(self, index):
        if not (0 <= index < self.image_list_widget.count()): return
        item = self.image_list_widget.item(index)
        txt_path = os.path.splitext(self.image_list[index])[0] + '.txt'
        if not os.path.exists(txt_path):
            item.setBackground(QBrush(QColor("transparent")))
            return
        try:
            with open(txt_path, 'r') as f: content = f.read().strip()
            if not content: item.setBackground(QBrush(QColor("#FFA500")))
            else: item.setBackground(QBrush(QColor("#90EE90")))
        except Exception as e:
            item.setBackground(QBrush(QColor("transparent")))
            logging.error(f"Could not read status for {txt_path}: {e}")
            
    def on_list_item_select(self, current_item, previous_item):
        if current_item is None: return
        row = self.image_list_widget.row(current_item)
        if row != self.current_index:
            if previous_item is not None and self.current_rect_item:
                self.save_annotation(silent=True)
            self.load_image(row)

    def load_image(self, index):
        if not (0 <= index < len(self.image_list)):
            if self.pixmap_item: self.scene.removeItem(self.pixmap_item)
            if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
            self.pixmap_item = None
            self.current_rect_item = None
            self.current_index = -1
            self.statusBar.showMessage("No images to display.")
            return

        if self.pixmap_item: self.scene.removeItem(self.pixmap_item)
        if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
        
        self.current_index = index
        self.image_list_widget.setCurrentRow(index)
        self.current_image_path = self.image_list[index]
        pixmap = QPixmap(self.current_image_path)

        if pixmap.isNull():
            QMessageBox.warning(self, "Load Error", f"Failed to load image: {self.current_image_path}")
            return

        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(-1)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.statusBar.showMessage(f"Image {index + 1}/{len(self.image_list)}: {os.path.basename(self.current_image_path)}")
        self.load_existing_annotation()

        if self.is_faiss_trained and not self.current_rect_item:
            self.auto_annotate(silent=True)

    def load_existing_annotation(self):
        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        if not os.path.exists(txt_path): return
        try:
            with open(txt_path, 'r') as f: data = f.readline().split()
            if len(data) < 5: return
            class_id, cx, cy, w, h = int(data[0]), *map(float, data[1:5])
            img_w, img_h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
            abs_w, abs_h = w * img_w, h * img_h
            x_min, y_min = cx * img_w - abs_w / 2, cy * img_h - abs_h / 2
            rect = QRectF(x_min, y_min, abs_w, abs_h)
            self.draw_bbox(rect, Qt.GlobalColor.red)
            if class_id < self.class_combo.count(): self.class_combo.setCurrentIndex(class_id)
        except Exception as e:
            logging.error(f"Failed to load annotation for {self.current_image_path}: {e}")

    def draw_bbox(self, rect, color):
        if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
        self.current_rect_item = QGraphicsRectItem(rect)
        self.current_rect_item.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
        self.scene.addItem(self.current_rect_item)

    def clear_annotation(self):
        if self.current_rect_item:
            self.scene.removeItem(self.current_rect_item)
            self.current_rect_item = None
        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        if os.path.exists(txt_path):
            try:
                os.remove(txt_path)
                self.statusBar.showMessage(f"Deleted annotation file: {os.path.basename(txt_path)}", 3000)
                logging.info(f"Deleted annotation file: {txt_path}")
                self.update_list_item_status(self.current_index)
            except OSError as e:
                QMessageBox.warning(self, "Delete Error", f"Could not delete annotation file:\n{e}")
                logging.error(f"Failed to delete {txt_path}: {e}")

    def scene_mouse_pressed(self, pos):
        if not self.annotation_mode: return
        self.start_point = pos
        rect = QRectF(self.start_point, self.start_point)
        self.draw_bbox(rect, Qt.GlobalColor.red)

    def scene_mouse_moved(self, pos):
        if self.annotation_mode:
            scene_rect = self.scene.sceneRect()
            self.crosshair_h.setLine(scene_rect.left(), pos.y(), scene_rect.right(), pos.y())
            self.crosshair_v.setLine(pos.x(), scene_rect.top(), pos.x(), scene_rect.bottom())
        if self.current_rect_item and self.annotation_mode and not self.start_point.isNull():
            rect = QRectF(self.start_point, pos).normalized()
            self.current_rect_item.setRect(rect)

    def scene_mouse_released(self, pos):
        if not self.annotation_mode or not self.current_rect_item: return
        rect = QRectF(self.start_point, pos).normalized()
        self.current_rect_item.setRect(rect)
        self.start_point = QPointF()

    def next_image(self):
        if self.current_rect_item: self.save_annotation(silent=True)
        if self.current_index < len(self.image_list) - 1: self.load_image(self.current_index + 1)

    def prev_image(self):
        if self.current_rect_item: self.save_annotation(silent=True)
        if self.current_index > 0: self.load_image(self.current_index - 1)
            
    def toggle_annotation_mode(self):
        self.annotation_mode = not self.annotation_mode
        cursor = Qt.CursorShape.CrossCursor if self.annotation_mode else Qt.CursorShape.ArrowCursor
        self.view.setCursor(QCursor(cursor))
        self.toggle_crosshairs(self.annotation_mode)
        status = "ON" if self.annotation_mode else "OFF"
        self.statusBar.showMessage(f"Annotation Mode: {status}", 2000)
        logging.info(f"Annotation mode toggled to {status}.")

    def toggle_crosshairs(self, visible):
        if self.crosshair_h: self.crosshair_h.setVisible(visible)
        if self.crosshair_v: self.crosshair_v.setVisible(visible)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnnotationApp()
    window.show()
    sys.exit(app.exec())