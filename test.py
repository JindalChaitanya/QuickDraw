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
    QListWidget, QListWidgetItem, QCompleter, QLabel, QCheckBox
)
from PyQt6.QtGui import QPixmap, QPen, QPainter, QColor, QBrush, QCursor, QAction
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QStringListModel

# GMS is in the xfeatures2d module in the contrib package
from cv2.xfeatures2d import matchGMS

logging.basicConfig(
    filename='annotation_tool.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AnnotationScene(QGraphicsScene):
    # (Unchanged)
    mousePressed = pyqtSignal(QPointF)
    mouseMoved = pyqtSignal(QPointF)
    mouseReleased = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.mousePressed.emit(event.scenePos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.mouseReleased.emit(event.scenePos())
        super().mouseReleaseEvent(event)


class AnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuickDraw (AKAZE + GMS + MAGSAC)")
        self.setGeometry(100, 100, 1200, 800)

        # --- UPGRADED FEATURE DETECTION AND PROCESSING ---
        # OPTION 1: Using AKAZE instead of SIFT. It's often more robust and faster.
        self.detector = cv2.AKAZE_create()
        self.max_features_per_image = 800 # Can afford a few more features
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        # --- PER-CLASS FAISS INDEX SETUP ---
        # CRITICAL: AKAZE descriptors are 61 bytes, not 128 like SIFT.
        self.descriptor_dimension = 61
        self.min_refs_for_training = 10
        self.faiss_indices = {}
        self.is_class_trained = {}
        self.references = []
        self.faiss_mappers = {}

        # --- UI & STATE ---
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
        logging.info("Application initialized with AKAZE, GMS, and MAGSAC.")

    def init_ui(self):
        # (This UI code is unchanged from the previous version)
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
        bottom_controls_layout.addWidget(self.clear_btn)
        self.auto_btn = QPushButton("Auto-Annotate")
        self.auto_btn.setEnabled(False)
        bottom_controls_layout.addWidget(self.auto_btn)
        self.auto_run_checkbox = QCheckBox("Auto-run on Next Image")
        self.auto_run_checkbox.setChecked(True)
        bottom_controls_layout.addWidget(self.auto_run_checkbox)
        bottom_controls_layout.addStretch()
        self.save_btn = QPushButton("Save (Ctrl+S)")
        bottom_controls_layout.addWidget(self.save_btn)
        self.delete_btn = QPushButton("Delete Image")
        self.delete_btn.setStyleSheet("background-color: #FF6347; color: white;")
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
        self.delete_btn.clicked.connect(self.delete_current_image)
        self.clear_btn.clicked.connect(self.clear_annotation)
        self.next_btn.clicked.connect(self.next_image)
        self.prev_btn.clicked.connect(self.prev_image)
        self.scene.mousePressed.connect(self.scene_mouse_pressed)
        self.scene.mouseMoved.connect(self.scene_mouse_moved)
        self.scene.mouseReleased.connect(self.scene_mouse_released)
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_annotation)
        self.addAction(save_action)

    def _detect_and_prune_features(self, gray_image):
        # OPTION 1: Using the AKAZE detector
        kp, des = self.detector.detectAndCompute(gray_image, None)
        if kp is None or len(kp) == 0: return None, None
        
        # Pruning is still useful to keep descriptor count uniform
        kp_des_pairs = sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)
        if len(kp_des_pairs) > self.max_features_per_image:
            kp_des_pairs = kp_des_pairs[:self.max_features_per_image]
        
        # Unzip after sorting and pruning
        kp, des = zip(*kp_des_pairs)
        return kp, np.array(des)

    def auto_annotate(self, silent=False):
        class_id = self.class_combo.currentIndex()
        class_name = self.class_combo.currentText()

        if not self.is_class_trained.get(class_id, False):
            msg = f"Not enough references for class '{class_name}' to auto-annotate."
            if not silent: QMessageBox.information(self, "Auto-Annotate", msg)
            else: self.statusBar.showMessage(msg, 4000)
            return

        self.statusBar.showMessage(f"Searching for '{class_name}'...", 0)
        QApplication.processEvents()

        index = self.faiss_indices[class_id]
        mapper = self.faiss_mappers[class_id]
        current_img_bgr = cv2.imread(self.current_image_path)
        if current_img_bgr is None: return
        
        gray_enhanced = self.clahe.apply(cv2.cvtColor(current_img_bgr, cv2.COLOR_BGR2GRAY))
        kp1, des1 = self._detect_and_prune_features(gray_enhanced)
        
        if des1 is None:
            msg = "No strong features found in current image."
            if not silent: QMessageBox.warning(self, "Auto-Annotate", msg)
            else: self.statusBar.showMessage(msg, 4000)
            return

        # STAGE 1: FAISS SEARCH
        des1_float32 = np.float32(des1)
        distances, indices = index.search(des1_float32, 2)

        good_matches_by_candidate = {}
        for i in range(len(des1)):
            if distances[i][0] < 0.75 * distances[i][1]:
                map_info = mapper[indices[i][0]]
                # Just store the path. We'll do a full match later.
                good_matches_by_candidate.setdefault(map_info['ref']['image_path'], []).append(map_info['ref'])

        if not good_matches_by_candidate:
            msg = "No promising candidate images found via Faiss search."
            if not silent: QMessageBox.information(self, "Auto-Annotate", msg)
            else: self.statusBar.showMessage(msg, 4000)
            return
        
        # --- NEW PIPELINE: FLANN -> GMS -> MAGSAC ---
        sorted_candidates = sorted(good_matches_by_candidate.items(), key=lambda item: len(item[1]), reverse=True)
        best_match = {'ref': None, 'score': -1, 'M': None}
        MIN_INLIER_COUNT = 15
        RANSAC_THRESH = 5.0

        for ref_path, refs in sorted_candidates[:10]: # Check top 10 candidates
            ref = refs[0] # All refs with this path are the same object
            
            # STAGE 2: FULL FLANN MATCH
            ref_des_float32 = np.float32(ref['des'])
            matches = self.flann.knnMatch(des1_float32, ref_des_float32, k=2)
            
            # Apply Lowe's Ratio Test to get raw good matches
            raw_good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    raw_good_matches.append(m)

            # STAGE 3: GMS FILTERING
            # This is a very fast and effective way to remove geometric outliers
            if len(raw_good_matches) < 20: continue # GMS needs a few matches to work
            
            gms_matches = matchGMS(gray_enhanced.shape[:2], ref['bbox_shape'], kp1, ref['kp'], raw_good_matches, withScale=False, withRotation=False, thresholdFactor=6.0)

            if len(gms_matches) < MIN_INLIER_COUNT: continue

            # STAGE 4: GEOMETRIC VERIFICATION WITH MAGSAC++
            src_pts = np.float32([kp1[m.queryIdx].pt for m in gms_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([ref['kp'][m.trainIdx].pt for m in gms_matches]).reshape(-1, 1, 2)
            
            # OPTION 2: Use USAC_MAGSAC for superior outlier rejection
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.USAC_MAGSAC, RANSAC_THRESH)

            if M is not None:
                inlier_count = np.sum(mask)
                if inlier_count > best_match['score']:
                    best_match.update({'score': inlier_count, 'ref': ref, 'M': M})
        
        if best_match['ref'] is None:
            msg = f"Failed to find a robust geometric match. Best score: {best_match['score']}"
            if not silent: QMessageBox.information(self, "Auto-Annotate", msg)
            else: self.statusBar.showMessage(msg, 4000)
            return
        
        # Found a match, now draw it
        ref = best_match['ref']
        M = best_match['M']
        ref_bbox = ref['bbox']
        ref_pts = np.float32([[ref_bbox[0], ref_bbox[1]], [ref_bbox[0], ref_bbox[3]],
                              [ref_bbox[2], ref_bbox[3]], [ref_bbox[2], ref_bbox[1]]]).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(ref_pts, M)
        x_min, y_min, w, h = cv2.boundingRect(transformed_pts)
        new_rect = QRectF(x_min, y_min, w, h)
        
        self.draw_bbox(new_rect, Qt.GlobalColor.blue)
        self.class_combo.setCurrentIndex(ref['class_id'])
        self.statusBar.showMessage(f"Match found with {best_match['score']} inliers.", 5000)

    # --- Most other methods are unchanged, but _build_initial_references needs to save bbox shape for GMS ---

    def _build_initial_references(self):
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

                gray_enhanced = self.clahe.apply(img)
                kp, des = self._detect_and_prune_features(gray_enhanced)
                if des is not None and len(des) > 0:
                    h, w = img.shape[:2]
                    cx, cy, bw, bh = map(float, data[1:5])
                    x_min, y_min = (cx - bw / 2) * w, (cy - bh / 2) * h
                    x_max, y_max = (cx + bw / 2) * w, (cy + bh / 2) * h
                    ref = {
                        'image_path': img_path, 'kp': kp, 'des': des, 
                        'bbox': [x_min, y_min, x_max, y_max], 
                        'class_id': class_id,
                        # Add image shape for GMS matching
                        'bbox_shape': (h, w) 
                    }
                    self.references.append(ref)
            except Exception as e:
                logging.error(f"Error loading reference for {img_path}: {e}")
        
        self._rebuild_all_faiss_indices() # This function remains the same internally
        self.statusBar.showMessage(f"Loaded {len(self.references)} references into {len(self.faiss_indices)} class indices.", 5000)

    def _update_index_with_new_annotation(self, rect, class_id):
        # (Slight modification to add bbox_shape)
        self.references = [ref for ref in self.references if ref['image_path'] != self.current_image_path]
        img_gray = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: return

        gray_enhanced = self.clahe.apply(img_gray)
        kp, des = self._detect_and_prune_features(gray_enhanced)
        if des is None: return

        new_ref = {
            'image_path': self.current_image_path, 'kp': kp, 'des': des,
            'bbox': [rect.left(), rect.top(), rect.right(), rect.bottom()],
            'class_id': class_id,
            'bbox_shape': gray_enhanced.shape[:2] # Add shape
        }
        self.references.append(new_ref)
        
        class_refs = [ref for ref in self.references if ref['class_id'] == class_id]
        if not self.is_class_trained.get(class_id, False) and len(class_refs) >= self.min_refs_for_training:
            self._build_class_index(class_id, class_refs)
        elif self.is_class_trained.get(class_id, False):
            index = self.faiss_indices[class_id]
            mapper = self.faiss_mappers[class_id]
            des_float32 = np.float32(des)
            index.add(des_float32)
            for kp_idx, _ in enumerate(des):
                mapper.append({'ref': new_ref, 'kp_idx': kp_idx})
        
        self.update_auto_annotate_button_status()

    # ===================================================================
    # ALL REMAINING METHODS BELOW ARE UNCHANGED FROM THE PREVIOUS VERSION
    # ===================================================================
    def _build_class_index(self, class_id, class_refs):
        if len(class_refs) < self.min_refs_for_training:
            self.is_class_trained[class_id] = False
            return
        all_descriptors_list = []
        mapper = []
        for ref in class_refs:
            for kp_idx, des in enumerate(ref['des']):
                mapper.append({'ref': ref, 'kp_idx': kp_idx})
                all_descriptors_list.append(des)
        all_descriptors = np.array(all_descriptors_list).astype('float32')
        num_descriptors = all_descriptors.shape[0]
        if num_descriptors < self.min_refs_for_training * 5:
            self.is_class_trained[class_id] = False
            return
        nlist = int(min(1024, 8 * np.sqrt(num_descriptors)))
        nlist = max(4, nlist)
        quantizer = faiss.IndexFlatL2(self.descriptor_dimension)
        index = faiss.IndexIVFFlat(quantizer, self.descriptor_dimension, nlist)
        index.nprobe = max(1, nlist // 8)
        index.train(all_descriptors)
        index.add(all_descriptors)
        self.faiss_indices[class_id] = index
        self.faiss_mappers[class_id] = mapper
        self.is_class_trained[class_id] = True
        logging.info(f"Built index for class {class_id} with {index.ntotal} vectors. (nlist={nlist}, nprobe={index.nprobe})")

    def _rebuild_all_faiss_indices(self):
        self.faiss_indices.clear(); self.faiss_mappers.clear(); self.is_class_trained.clear()
        self.statusBar.showMessage("Building Faiss indices...", 0)
        QApplication.processEvents()
        refs_by_class = {}
        for ref in self.references:
            refs_by_class.setdefault(ref['class_id'], []).append(ref)
        for class_id, class_refs in refs_by_class.items():
            self._build_class_index(class_id, class_refs)
        self.update_auto_annotate_button_status()
        self.statusBar.showMessage(f"Finished building {len(self.faiss_indices)} class indices.", 3000)

    def save_annotation(self):
        if not self.current_rect_item or self.class_combo.currentIndex() < 0:
            QMessageBox.warning(self, "Save Error", "Please draw a bounding box and select a class.")
            return
        rect = self.current_rect_item.rect()
        img_w, img_h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
        if img_w == 0 or img_h == 0: return
        cx,cy,w,h = (rect.left()+rect.width()/2)/img_w, (rect.top()+rect.height()/2)/img_h, rect.width()/img_w, rect.height()/img_h
        class_id = self.class_combo.currentIndex()
        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        with open(txt_path, 'w') as f: f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        self._update_index_with_new_annotation(rect, class_id)
        self.update_list_item_status(self.current_index)
        self.draw_bbox(rect, Qt.GlobalColor.red)
        self.statusBar.showMessage(f"Annotation saved and index updated.", 3000)
        logging.info(f"Annotation saved for {self.current_image_path}")

    def delete_current_image(self):
        if self.current_index == -1: return
        img_path = self.current_image_path
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        reply = QMessageBox.question(self, 'Confirm Deletion', 
                                     f"Are you sure you want to permanently delete this image and its annotation?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if os.path.exists(img_path): os.remove(img_path)
                if os.path.exists(txt_path): os.remove(txt_path)
            except OSError as e:
                QMessageBox.warning(self, "Delete Error", f"Error deleting files:\n{e}")
                return
            deleted_index = self.current_index
            self.image_list.pop(deleted_index)
            self.image_list_widget.takeItem(deleted_index)
            was_ref = any(ref['image_path'] == img_path for ref in self.references)
            self.references = [ref for ref in self.references if ref['image_path'] != img_path]
            if was_ref: self._rebuild_all_faiss_indices()
            if not self.image_list: self.load_image(-1)
            else: self.load_image(min(deleted_index, len(self.image_list) - 1))

    def update_auto_annotate_button_status(self):
        num_trained = sum(1 for is_trained in self.is_class_trained.values() if is_trained)
        if num_trained > 0:
            self.auto_btn.setEnabled(True)
            self.auto_btn.setText(f"Auto-Annotate ({num_trained} classes ready)")
        else:
            self.auto_btn.setEnabled(False)
            self.auto_btn.setText(f"Auto-Annotate (needs {self.min_refs_for_training} refs)")
    
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder: return
        logging.info(f"Opening folder: {folder}")
        self.path_label.setText(folder)
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_list = []
        for ext in image_extensions: self.image_list.extend(glob.glob(os.path.join(folder, ext)))
        self.image_list = sorted([p.replace('\\', '/') for p in self.image_list])
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "No images found in the selected folder.")
            return
        classes_path = os.path.join(folder, 'classes.txt')
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            self.class_combo.clear()
            self.class_combo.addItems(classes)
            self.completer.setModel(QStringListModel(classes, self))
        else:
            QMessageBox.warning(self, "File Not Found", "'classes.txt' not found in folder. Class list will be empty.")
            self.class_combo.clear()
        self.populate_image_list_panel()
        self._build_initial_references()
        if self.image_list: self.load_image(0)
            
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
            item.setBackground(QBrush(Qt.GlobalColor.transparent))
            return
        try:
            with open(txt_path, 'r') as f: content = f.read().strip()
            if not content: item.setBackground(QBrush(QColor("#FFA500")))
            else: item.setBackground(QBrush(QColor("#90EE90")))
        except Exception:
            item.setBackground(QBrush(Qt.GlobalColor.transparent))
            
    def on_list_item_select(self, current_item, previous_item):
        if current_item is None: return
        row = self.image_list_widget.row(current_item)
        if row != self.current_index: self.load_image(row)

    def load_image(self, index):
        if not (0 <= index < len(self.image_list)):
            if self.pixmap_item: self.scene.removeItem(self.pixmap_item)
            if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
            self.pixmap_item, self.current_rect_item, self.current_index = None, None, -1
            self.statusBar.showMessage("No images to display.")
            return
        if self.pixmap_item: self.scene.removeItem(self.pixmap_item)
        self.clear_annotation(save_and_rebuild=False)
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
        if self.auto_run_checkbox.isChecked() and not self.current_rect_item:
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
        pen_width = max(2, int(self.view.transform().m11() * 2))
        self.current_rect_item.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine))
        self.scene.addItem(self.current_rect_item)

    def clear_annotation(self, save_and_rebuild=True):
        if self.current_rect_item:
            self.scene.removeItem(self.current_rect_item)
            self.current_rect_item = None
        if not save_and_rebuild: return
        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        if os.path.exists(txt_path):
            try:
                os.remove(txt_path)
                self.statusBar.showMessage(f"Deleted annotation file: {os.path.basename(txt_path)}", 3000)
                self.update_list_item_status(self.current_index)
                self.references = [ref for ref in self.references if ref['image_path'] != self.current_image_path]
                self._rebuild_all_faiss_indices()
            except OSError as e:
                QMessageBox.warning(self, "Delete Error", f"Could not delete annotation file:\n{e}")

    def scene_mouse_pressed(self, pos):
        if not self.annotation_mode: return
        self.start_point = pos
        self.draw_bbox(QRectF(pos, pos), Qt.GlobalColor.red)

    def scene_mouse_moved(self, pos):
        if self.annotation_mode:
            scene_rect = self.scene.sceneRect()
            self.crosshair_h.setLine(scene_rect.left(), pos.y(), scene_rect.right(), pos.y())
            self.crosshair_v.setLine(pos.x(), scene_rect.top(), pos.x(), scene_rect.bottom())
        if self.current_rect_item and self.annotation_mode and not self.start_point.isNull():
            self.current_rect_item.setRect(QRectF(self.start_point, pos).normalized())

    def scene_mouse_released(self, pos):
        if not self.annotation_mode or not self.current_rect_item: return
        self.current_rect_item.setRect(QRectF(self.start_point, pos).normalized())
        self.start_point = QPointF()

    def next_image(self):
        if self.current_rect_item: self.save_annotation()
        if self.current_index < len(self.image_list) - 1: self.load_image(self.current_index + 1)

    def prev_image(self):
        if self.current_rect_item: self.save_annotation()
        if self.current_index > 0: self.load_image(self.current_index - 1)
            
    def toggle_annotation_mode(self):
        self.annotation_mode = not self.annotation_mode
        cursor = Qt.CursorShape.CrossCursor if self.annotation_mode else Qt.CursorShape.ArrowCursor
        self.view.setCursor(QCursor(cursor))
        self.toggle_crosshairs(self.annotation_mode)
        status = "ON" if self.annotation_mode else "OFF"
        self.statusBar.showMessage(f"Annotation Mode (W): {status}", 2000)

    def toggle_crosshairs(self, visible):
        self.crosshair_h.setVisible(visible)
        self.crosshair_v.setVisible(visible)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item: self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnnotationApp()
    window.show()
    sys.exit(app.exec())