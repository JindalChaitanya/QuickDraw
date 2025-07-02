import sys
import os
import glob
import cv2
import numpy as np
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QGraphicsView, QGraphicsScene, QFileDialog,
    QMessageBox, QStatusBar, QGraphicsRectItem, QGraphicsPixmapItem,
    QListWidget, QListWidgetItem, QCompleter, QLabel, QCheckBox
)
from PyQt6.QtGui import QPixmap, QPen, QPainter, QColor, QBrush, QCursor
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QStringListModel, QObject, QThread

# --- Setup Logging ---
logging.basicConfig(
    filename='annotation_tool.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- AnnotationScene ---
# A QGraphicsScene subclass that emits signals for mouse events.
class AnnotationScene(QGraphicsScene):
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


# --- AnnotationWorker ---
# This object runs all heavy computations in a separate thread.
class AnnotationWorker(QObject):
    result_ready = pyqtSignal(QRectF, int, str)
    no_match_found = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.sift = cv2.SIFT_create(nfeatures=2000)

    def run_auto_annotate(self, current_image_path, references, all_descriptors, descriptor_map, global_flann):
        try:
            if all_descriptors is None or len(all_descriptors) == 0:
                self.no_match_found.emit("Global feature index is not built.")
                return

            current_img_bgr = cv2.imread(current_image_path)
            if current_img_bgr is None:
                self.error_occurred.emit(f"Failed to read image: {current_image_path}")
                return
            
            current_img_gray = cv2.cvtColor(current_img_bgr, cv2.COLOR_BGR2GRAY)
            kp1, des1 = self.sift.detectAndCompute(current_img_gray, None)
            
            if des1 is None:
                self.no_match_found.emit("No SIFT features found in the current image.")
                return

            # 1. Search the GLOBAL index
            all_matches = global_flann.knnMatch(des1, k=2)
            good_matches = [m for m, n in all_matches if m.distance < 0.75 * n.distance]
            if not good_matches:
                self.no_match_found.emit("No good matches found in the global index.")
                return

            # 2. Vote for the best reference image
            votes = np.zeros(len(references))
            for m in good_matches:
                ref_idx = descriptor_map[m.trainIdx]
                votes[ref_idx] += 1
            
            best_ref_idx = np.argmax(votes)
            best_ref = references[best_ref_idx]

            # 3. Perform geometric verification ONLY on the best candidate
            bf = cv2.BFMatcher()
            one_on_one_matches = bf.knnMatch(des1, best_ref['des'], k=2)
            good_one_on_one = [m for m, n in one_on_one_matches if m.distance < 0.75 * n.distance]

            MIN_MATCH_COUNT = 10
            if len(good_one_on_one) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_one_on_one]).reshape(-1, 1, 2)
                dst_pts = np.float32([best_ref['kp'][m.trainIdx].pt for m in good_one_on_one]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    ref_bbox = best_ref['bbox']
                    ref_pts = np.float32([[ref_bbox[0], ref_bbox[1]], [ref_bbox[0], ref_bbox[3]], [ref_bbox[2], ref_bbox[3]], [ref_bbox[2], ref_bbox[1]]]).reshape(-1, 1, 2)
                    transformed_pts = cv2.perspectiveTransform(ref_pts, M)
                    x_min, y_min = np.min(transformed_pts, axis=0)[0]
                    x_max, y_max = np.max(transformed_pts, axis=0)[0]
                    h, w = current_img_gray.shape
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    new_rect = QRectF(QPointF(x_min, y_min), QPointF(x_max, y_max))
                    self.result_ready.emit(new_rect, best_ref['class_id'], os.path.basename(best_ref['image_path']))
                else:
                    self.no_match_found.emit("Homography check failed for the best candidate.")
            else:
                self.no_match_found.emit(f"Not enough matches ({len(good_one_on_one)}/{MIN_MATCH_COUNT}) with best candidate to verify.")
        
        except Exception as e:
            logging.error(f"Error in annotation worker: {e}", exc_info=True)
            self.error_occurred.emit(f"An unexpected worker error occurred: {e}")


class AnnotationApp(QMainWindow):
    """The main application window for high-performance image annotation."""
    start_annotation_job = pyqtSignal(str, list, np.ndarray, list, cv2.FlannBasedMatcher)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # --- CV Components ---
        self.sift = cv2.SIFT_create(nfeatures=2000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        # --- Data & State ---
        self.references = []
        self.image_list = []
        self.current_index = -1
        self.current_image_path = None
        self.current_rect_item = None
        self.pixmap_item = None
        self.start_point = None
        self.is_annotating = False
        self.annotation_mode = False
        self.folder_path = None
        self.cache_file_name = "sift_features_cache.npz"

        # --- Fast Incremental Index Data ---
        self.all_descriptors_list = [] 
        self.descriptor_map = [] 
        self.global_flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.is_index_built = False
        
        self.init_ui()
        self.init_worker_thread()
        logging.info("Application initialized.")

    def init_ui(self):
        """Initializes the user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        overall_layout = QVBoxLayout(central_widget)

        # --- Top Controls ---
        top_controls_layout = QHBoxLayout()
        self.folder_btn = QPushButton("Select Folder")
        self.folder_btn.clicked.connect(self.open_folder)
        top_controls_layout.addWidget(self.folder_btn)
        
        self.path_label = QLabel("No folder selected.")
        self.path_label.setStyleSheet("font-style: italic; color: grey;")
        top_controls_layout.addWidget(self.path_label, 1)

        self.auto_pilot_checkbox = QCheckBox("Auto-Pilot Mode")
        self.auto_pilot_checkbox.setToolTip("When checked, automatically annotates new images and saves on 'Next'.")
        self.auto_pilot_checkbox.stateChanged.connect(self.on_autopilot_change)
        self.auto_pilot_checkbox.setEnabled(False)
        top_controls_layout.addWidget(self.auto_pilot_checkbox)
        
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

        # --- Main Content ---
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

        # --- Bottom Controls & Navigation ---
        bottom_controls_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear (Esc)")
        bottom_controls_layout.addWidget(self.clear_btn)
        self.auto_btn = QPushButton("Auto-Annotate (G)")
        self.auto_btn.setShortcut("G")
        self.save_btn = QPushButton("Save (S)")
        self.save_btn.setShortcut("S")
        self.delete_btn = QPushButton("Delete Image")
        self.delete_btn.setStyleSheet("background-color: #FF6347; color: white;")
        bottom_controls_layout.addWidget(self.auto_btn)
        bottom_controls_layout.addWidget(self.save_btn)
        bottom_controls_layout.addWidget(self.delete_btn)
        right_panel_layout.addLayout(bottom_controls_layout)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous (A)")
        self.prev_btn.setShortcut("A")
        self.next_btn = QPushButton("Next (D)")
        self.next_btn.setShortcut("D")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        right_panel_layout.addLayout(nav_layout)
        overall_layout.addLayout(main_content_layout)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # --- Connections ---
        self.auto_btn.clicked.connect(self.auto_annotate)
        self.save_btn.clicked.connect(self.save_annotation)
        self.delete_btn.clicked.connect(self.delete_current_image)
        self.clear_btn.clicked.connect(self.clear_annotation)
        self.next_btn.clicked.connect(self.next_image)
        self.prev_btn.clicked.connect(self.prev_image)
        self.scene.mousePressed.connect(self.scene_mouse_pressed)
        self.scene.mouseMoved.connect(self.scene_mouse_moved)
        self.scene.mouseReleased.connect(self.scene_mouse_released)

    def init_worker_thread(self):
        """Sets up the background worker and thread."""
        self.thread = QThread()
        self.worker = AnnotationWorker()
        self.worker.moveToThread(self.thread)
        self.worker.result_ready.connect(self.on_annotation_result)
        self.worker.no_match_found.connect(self.on_annotation_no_match)
        self.worker.error_occurred.connect(self.on_annotation_error)
        self.start_annotation_job.connect(self.worker.run_auto_annotate)
        self.thread.start()
        
    def open_folder(self):
        """Opens a folder, loads images and classes, and builds the feature index."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder: return
        self.folder_path = folder
        self.path_label.setText(folder)
        self.path_label.setStyleSheet("")
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_list = []
        for ext in image_extensions:
            self.image_list.extend(glob.glob(os.path.join(folder, ext)))
        self.image_list = sorted(self.image_list)

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
            QMessageBox.warning(self, "File Not Found", "classes.txt not found.")
            self.class_combo.clear()
        
        self.populate_image_list_panel()
        self.load_and_build_index()
        if self.image_list:
            self.load_image(0)
    
    def load_and_build_index(self):
        """Loads all features from a single master cache file for extreme speed."""
        self.statusBar.showMessage("Loading features and building index...")
        QApplication.processEvents()
        
        cache_path = os.path.join(self.folder_path, self.cache_file_name)
        self.references = []
        self.all_descriptors_list = []
        self.descriptor_map = []
        
        if os.path.exists(cache_path):
            try:
                cached_data = np.load(cache_path, allow_pickle=True)
                # Reconstruct lightweight references and full references for homography
                light_refs = cached_data['references'].tolist()
                descriptors = cached_data['descriptors']
                self.all_descriptors_list = [d for d in descriptors]
                self.descriptor_map = cached_data['map'].tolist()
                
                # Re-read keypoints for full reference objects needed for homography
                for i, ref_data in enumerate(light_refs):
                    img = cv2.imread(ref_data['image_path'], cv2.IMREAD_GRAYSCALE)
                    kp, _ = self.sift.detectAndCompute(img, None)
                    full_ref = ref_data.copy()
                    full_ref['kp'] = kp
                    full_ref['des'] = descriptors[i]
                    self.references.append(full_ref)
                self.statusBar.showMessage(f"Loaded {len(self.references)} references from cache.", 2000)
            except Exception as e:
                logging.error(f"Failed to load cache file {cache_path}: {e}. Rebuilding from scratch.")
                self.references = []

        if not self.references:
            for img_path in self.image_list:
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                if not os.path.exists(txt_path): continue
                self.add_single_reference(img_path, rebuild_index=False)
            self.save_master_cache()

        if self.all_descriptors_list:
            self.rebuild_flann_index_incrementally()
            self.auto_btn.setEnabled(True)
            self.auto_pilot_checkbox.setEnabled(True)
        else:
            self.statusBar.showMessage("No references found to build index.", 4000)
            self.auto_btn.setEnabled(False)
            self.auto_pilot_checkbox.setEnabled(False)

    def add_single_reference(self, img_path, rebuild_index=True):
        """Computes SIFT for a single image and adds it to the in-memory lists."""
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        try:
            with open(txt_path, 'r') as f: data = f.readline().split()
            if len(data) < 5: return
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return
            kp, des = self.sift.detectAndCompute(img, None)
            
            if des is not None:
                class_id = int(data[0])
                h, w = img.shape[:2]
                cx, cy, bw, bh = map(float, data[1:5])
                x_min, y_min = (cx - bw / 2) * w, (cy - bh / 2) * h
                x_max, y_max = (cx + bw / 2) * w, (cy + bh / 2) * h
                
                ref_for_homography = {
                    'image_path': img_path, 'bbox': [x_min, y_min, x_max, y_max], 
                    'class_id': class_id, 'kp': kp, 'des': des
                }

                self.references.append(ref_for_homography)
                self.all_descriptors_list.append(des)
                self.descriptor_map.extend([len(self.references) - 1] * len(des))

                if rebuild_index and self.all_descriptors_list:
                    self.rebuild_flann_index_incrementally()
        except Exception as e:
            logging.error(f"Failed to add reference for {img_path}: {e}")

    def rebuild_flann_index_incrementally(self):
        """Rebuilds the FLANN index from the current in-memory descriptor list."""
        self.statusBar.showMessage("Updating feature index...")
        all_des = np.vstack(self.all_descriptors_list).astype(np.float32)
        self.global_flann.clear()
        self.global_flann.add([all_des])
        self.global_flann.train()
        self.is_index_built = True
        self.statusBar.showMessage(f"Index updated. Total features: {len(all_des)}", 2000)

    def save_master_cache(self):
        """Saves all references and features to a single compressed file."""
        if not self.folder_path or not self.references: return
        cache_path = os.path.join(self.folder_path, self.cache_file_name)
        
        light_references = [{'image_path': r['image_path'], 'bbox': r['bbox'], 'class_id': r['class_id']} for r in self.references]
        descriptors_to_save = [r['des'] for r in self.references]
        
        try:
            np.savez_compressed(cache_path, 
                                references=np.array(light_references, dtype=object),
                                descriptors=np.array(descriptors_to_save, dtype=object),
                                map=np.array(self.descriptor_map))
            logging.info(f"Master cache saved to {cache_path}")
        except Exception as e:
            logging.error(f"Could not save master cache: {e}")

    def save_annotation(self, silent=False):
        """Saves the current bounding box and updates the index if it's a new annotation."""
        if not self.current_rect_item or self.class_combo.currentIndex() < 0:
            if not silent: QMessageBox.warning(self, "Save Error", "Please draw a bounding box and select a class.")
            return False
            
        is_new_annotation = self.current_image_path not in [r['image_path'] for r in self.references]
        
        rect = self.current_rect_item.rect()
        img_w, img_h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
        if img_w == 0 or img_h == 0: return False
        
        cx = (rect.left() + rect.width() / 2) / img_w
        cy = (rect.top() + rect.height() / 2) / img_h
        w, h = rect.width() / img_w, rect.height() / img_h
        class_id = self.class_combo.currentIndex()
        
        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        if is_new_annotation:
            self.add_single_reference(self.current_image_path, rebuild_index=True)
            self.save_master_cache()
            
        self.update_list_item_status(self.current_index)
        self.draw_bbox(self.current_rect_item.rect(), Qt.GlobalColor.red)
        if not silent:
            self.statusBar.showMessage(f"Annotation saved to {os.path.basename(txt_path)}", 3000)
        logging.info(f"Annotation saved for {self.current_image_path}")
        return True

    def load_image(self, index):
        """Loads an image and triggers auto-annotation if in Auto-Pilot mode."""
        if not (0 <= index < len(self.image_list)):
            # Clear view if list is empty
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

        has_existing_annotation = os.path.exists(os.path.splitext(self.current_image_path)[0] + '.txt')
        if has_existing_annotation:
            self.load_existing_annotation()
        elif self.auto_pilot_checkbox.isChecked() and self.is_index_built:
            self.auto_annotate()

    def next_image(self):
        """Moves to the next image, saving automatically in Auto-Pilot mode."""
        if self.auto_pilot_checkbox.isChecked() and self.current_rect_item:
            self.save_annotation(silent=True)
        
        if self.current_index < len(self.image_list) - 1:
            self.load_image(self.current_index + 1)

    def prev_image(self):
        """Moves to the previous image, saving automatically in Auto-Pilot mode."""
        if self.auto_pilot_checkbox.isChecked() and self.current_rect_item:
            self.save_annotation(silent=True)

        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    # --- All other helper and event handler methods ---
    
    def on_autopilot_change(self, state):
        status = "ON" if state == Qt.CheckState.Checked.value else "OFF"
        self.statusBar.showMessage(f"Auto-Pilot Mode: {status}", 3000)

    def auto_annotate(self):
        if self.is_annotating:
            QMessageBox.warning(self, "Busy", "An auto-annotation job is already running.")
            return
        if not self.is_index_built:
            QMessageBox.information(self, "Auto-Annotate", "Feature index is not ready.")
            return
        self.is_annotating = True
        self.auto_btn.setEnabled(False)
        self.auto_btn.setText("Annotating...")
        self.statusBar.showMessage("Starting auto-annotation job...")
        all_des = np.vstack(self.all_descriptors_list).astype(np.float32) if self.all_descriptors_list else None
        self.start_annotation_job.emit(
            self.current_image_path, self.references, all_des, self.descriptor_map, self.global_flann
        )

    def on_annotation_result(self, rect, class_id, best_ref_path):
        self.draw_bbox(rect, Qt.GlobalColor.blue)
        if class_id < self.class_combo.count(): self.class_combo.setCurrentIndex(class_id)
        self.statusBar.showMessage(f"Success. Best match: {best_ref_path}", 5000)
        self.is_annotating = False
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("Auto-Annotate (G)")

    def on_annotation_no_match(self, message):
        self.statusBar.showMessage(f"Auto-annotation failed: {message}", 5000)
        self.is_annotating = False
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("Auto-Annotate (G)")

    def on_annotation_error(self, message):
        QMessageBox.critical(self, "Worker Error", message)
        self.statusBar.showMessage(f"Error during annotation: {message}", 5000)
        self.is_annotating = False
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("Auto-Annotate (G)")

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_W: self.toggle_annotation_mode()
        # Shortcuts for A, D, S, G are handled by QShortcut
        elif key == Qt.Key.Key_Escape: self.clear_annotation()
        else: super().keyPressEvent(event)

    def populate_image_list_panel(self):
        self.image_list_widget.clear()
        for img_path in self.image_list:
            item = QListWidgetItem(os.path.basename(img_path))
            self.image_list_widget.addItem(item)
        for i in range(self.image_list_widget.count()):
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
            if not content: item.setBackground(QBrush(QColor("#FFA500"))) # Orange for empty
            else: item.setBackground(QBrush(QColor("#90EE90"))) # Green for annotated
        except Exception as e:
            item.setBackground(QBrush(QColor("transparent")))
            logging.error(f"Could not read status for {txt_path}: {e}")

    def on_list_item_select(self, current_item, previous_item):
        if current_item is None: return
        row = self.image_list_widget.row(current_item)
        if row != self.current_index:
            if previous_item is not None and self.current_rect_item and self.auto_pilot_checkbox.isChecked():
                self.save_annotation(silent=True)
            self.load_image(row)

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
            if class_id < self.class_combo.count():
                self.class_combo.setCurrentIndex(class_id)
        except Exception as e:
            logging.error(f"Failed to load annotation for {self.current_image_path}: {e}")

    def draw_bbox(self, rect, color):
        if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
        self.current_rect_item = QGraphicsRectItem(rect)
        self.current_rect_item.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
        self.scene.addItem(self.current_rect_item)

    def clear_annotation(self):
        if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
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
        if self.current_rect_item and self.annotation_mode and self.start_point:
            self.current_rect_item.setRect(QRectF(self.start_point, pos).normalized())

    def scene_mouse_released(self, pos):
        if not self.annotation_mode or not self.current_rect_item: return
        if self.start_point:
            self.current_rect_item.setRect(QRectF(self.start_point, pos).normalized())
            self.start_point = None

    def toggle_annotation_mode(self):
        self.annotation_mode = not self.annotation_mode
        self.view.setCursor(QCursor(Qt.CursorShape.CrossCursor if self.annotation_mode else Qt.CursorShape.ArrowCursor))
        self.toggle_crosshairs(self.annotation_mode)
        status = "ON" if self.annotation_mode else "OFF"
        self.statusBar.showMessage(f"Annotation Mode: {status}", 2000)

    def toggle_crosshairs(self, visible):
        if self.crosshair_h: self.crosshair_h.setVisible(visible)
        if self.crosshair_v: self.crosshair_v.setVisible(visible)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def delete_current_image(self):
        if self.current_index == -1: return
        img_path = self.current_image_path
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        reply = QMessageBox.question(self, 'Confirm Deletion', f"Are you sure you want to permanently delete:\n- {os.path.basename(img_path)}\n- {os.path.basename(txt_path)} (if exists)", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(img_path)
                logging.info(f"Deleted image file: {img_path}")
                if os.path.exists(txt_path): os.remove(txt_path)
            except OSError as e:
                QMessageBox.warning(self, "Delete Error", f"Error deleting files:\n{e}")
                return
            
            deleted_index = self.current_index
            self.image_list.pop(deleted_index)
            self.image_list_widget.takeItem(deleted_index)
            
            # Deletion is complex, so a full rebuild is the safest strategy here
            self.load_and_build_index()
            
            if not self.image_list: self.load_image(-1)
            else: self.load_image(min(deleted_index, len(self.image_list) - 1))
            self.save_master_cache()

    def closeEvent(self, event):
        """Ensures the background thread is properly shut down on exit."""
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnnotationApp()
    window.show()
    sys.exit(app.exec())