import sys
import os
import glob
import cv2
import numpy as np
import logging
import h5py

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

# --- NEW HDF5 LOGIC ---
def _keypoints_to_array(kp):
    """Converts a list of cv2.KeyPoint objects to a NumPy array."""
    return np.array([[p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id] for p in kp])

def _array_to_keypoints(arr):
    """Converts a NumPy array back to a list of cv2.KeyPoint objects."""
    return [cv2.KeyPoint(x=p[0], y=p[1], _size=p[2], _angle=p[3], _response=p[4], _octave=int(p[5]), _class_id=int(p[6])) for p in arr]

def _sanitize_path_for_h5(path):
    """Sanitizes a path to be a valid HDF5 group name."""
    return path.replace('/', '_').replace('\\', '_').replace('.', '_')
# --- END NEW HDF5 LOGIC ---


class AnnotationScene(QGraphicsScene):
    # ... (No changes here)
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
        self.setWindowTitle("QuickDraw - Smart Reference Annotator (HDF5 Edition)")
        self.setGeometry(100, 100, 1200, 800)

        self.sift = cv2.SIFT_create(nfeatures=2000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.references = [] # This will now be a cache loaded from HDF5
        self.image_list = []
        self.current_index = -1
        self.current_image_path = None
        self.current_rect_item = None
        self.start_point = QPointF()
        self.pixmap_item = None
        
        self.annotation_mode = False
        self.manual_annotation_count = 0
        self.crosshair_h = None
        self.crosshair_v = None

        # --- MODIFIED FOR HDF5 ---
        self.h5_ref_path = None # Path to the HDF5 reference file
        self.current_annotation_is_manual = False
        # --- END MODIFIED ---

        self.init_ui()
        logging.info("Application initialized with SIFT/FLANN and HDF5 Reference Storage.")

    def init_ui(self):
        # ... (No changes in this function)
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


    def keyPressEvent(self, event):
        # ... (No changes here)
        key = event.key()
        if key == Qt.Key.Key_W: self.toggle_annotation_mode()
        elif key == Qt.Key.Key_A: self.prev_image()
        elif key == Qt.Key.Key_D: self.next_image()
        elif key == Qt.Key.Key_Escape: self.clear_annotation()
        else: super().keyPressEvent(event)


    def open_folder(self):
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

        # --- MODIFIED FOR HDF5 ---
        self.h5_ref_path = os.path.join(folder, 'references.h5')
        self.load_references_from_h5()
        # --- END MODIFIED ---

        self.populate_image_list_panel()
        if self.image_list:
            self.current_index = 0
            self.load_image(self.current_index)

    # --- REWRITTEN FOR HDF5 ---
    def load_references_from_h5(self):
        """Loads all reference data from the HDF5 file into the in-memory cache."""
        self.references = []
        if not self.h5_ref_path or not os.path.exists(self.h5_ref_path):
            self.update_auto_annotate_button_status()
            return

        self.statusBar.showMessage("Loading references from HDF5 store...")
        try:
            with h5py.File(self.h5_ref_path, 'r') as hf:
                image_folder = os.path.dirname(self.image_list[0]) if self.image_list else '.'
                for group_name in hf.keys():
                    group = hf[group_name]
                    # The image path is stored relative to the folder
                    img_path = os.path.join(image_folder, group.attrs['image_path'])
                    
                    # If image file doesn't exist anymore, skip this reference
                    if not os.path.exists(img_path):
                        logging.warning(f"Reference image {img_path} not found. Skipping.")
                        continue

                    kp_arr = group['keypoints'][:]
                    des = group['descriptors'][:]
                    bbox = group['bbox'][:]
                    class_id = group['class_id'][()]

                    ref = {
                        'image_path': img_path,
                        'kp': _array_to_keypoints(kp_arr),
                        'des': des,
                        'bbox': list(bbox),
                        'class_id': class_id
                    }
                    self.references.append(ref)
        except Exception as e:
            logging.error(f"Failed to load HDF5 reference file: {e}")
            QMessageBox.critical(self, "HDF5 Load Error", f"Could not read references.h5:\n{e}")

        self.manual_annotation_count = len(self.references)
        self.update_auto_annotate_button_status()
        self.statusBar.showMessage(f"Loaded {len(self.references)} references from HDF5.", 4000)

    # ... scene_mouse_released, auto_annotate remain the same ...
    def scene_mouse_released(self, pos):
        if not self.annotation_mode or not self.current_rect_item: return
        rect = QRectF(self.start_point, pos).normalized()
        self.current_rect_item.setRect(rect)
        self.start_point = QPointF()
        self.current_annotation_is_manual = True

    def auto_annotate(self, silent=False):
        if not self.references:
            if not silent: QMessageBox.information(self, "Auto-Annotate", "No manual references available. Please annotate a few images yourself first.")
            return

        current_img_bgr = cv2.imread(self.current_image_path)
        if current_img_bgr is None: return
        current_img_gray = cv2.cvtColor(current_img_bgr, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(current_img_gray, None)
        if des1 is None:
            if not silent: QMessageBox.warning(self, "Auto-Annotate", "Could not find SIFT features in the current image.")
            return

        best_match = {'ref': None, 'score': -1, 'M': None}
        MIN_MATCH_COUNT = 10

        for ref in self.references:
            if ref['des'] is None or len(ref['des']) < 2: continue
            try:
                all_matches = self.flann.knnMatch(des1, ref['des'], k=2)
            except cv2.error: continue
                
            good_matches = [m for m, n in all_matches if m.distance < 0.75 * n.distance]

            if len(good_matches) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([ref['kp'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                if M is not None and (inlier_count := np.sum(mask)) > best_match['score']:
                    best_match = {'score': inlier_count, 'ref': ref, 'M': M}
        
        if best_match['ref'] is None or best_match['score'] < MIN_MATCH_COUNT:
            if not silent:
                QMessageBox.information(self, "Auto-Annotate", f"Failed to find a geometrically consistent match. Best inlier count: {best_match['score']}")
            return

        ref = best_match['ref']
        M = best_match['M']
        ref_bbox = ref['bbox']
        ref_pts = np.float32([[ref_bbox[0], ref_bbox[1]], [ref_bbox[0], ref_bbox[3]], [ref_bbox[2], ref_bbox[3]], [ref_bbox[2], ref_bbox[1]]]).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(ref_pts, M)
        x_min, y_min = np.min(transformed_pts, axis=0)[0]
        x_max, y_max = np.max(transformed_pts, axis=0)[0]
        h, w = current_img_gray.shape
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)
        
        new_rect = QRectF(QPointF(x_min, y_min), QPointF(x_max, y_max))
        self.draw_bbox(new_rect, Qt.GlobalColor.blue) # Blue for auto-annotations
        if ref['class_id'] < self.class_combo.count():
            self.class_combo.setCurrentIndex(ref['class_id'])
        
        self.current_annotation_is_manual = False
        logging.info(f"Auto-annotation successful for {self.current_image_path} with {best_match['score']} inliers.")

    # --- MODIFIED FOR HDF5 ---
    def save_annotation(self, silent=False):
        if not self.current_rect_item or self.class_combo.currentIndex() < 0:
            if not silent: QMessageBox.warning(self, "Save Error", "Please draw a bounding box and select a class.")
            return

        rect = self.current_rect_item.rect()
        img_w, img_h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
        if img_w == 0 or img_h == 0: return

        cx, cy = (rect.left() + rect.width() / 2) / img_w, (rect.top() + rect.height() / 2) / img_h
        w, h = rect.width() / img_w, rect.height() / img_h
        class_id = self.class_combo.currentIndex()

        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        with open(txt_path, 'w') as f: f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Update the HDF5 reference store if this was a manual annotation
        if self.current_annotation_is_manual:
            self.update_h5_reference_store()
        
        self.current_annotation_is_manual = False

        self.update_list_item_status(self.current_index)
        self.draw_bbox(self.current_rect_item.rect(), Qt.GlobalColor.red)
        if not silent: 
            self.statusBar.showMessage(f"Annotation saved to {os.path.basename(txt_path)}", 3000)
        logging.info(f"Annotation saved for {self.current_image_path}")

    # --- REWRITTEN FOR HDF5 ---
    def update_h5_reference_store(self):
        """Adds or updates an entry for the current image in the HDF5 store."""
        img_gray = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: return
        
        kp, des = self.sift.detectAndCompute(img_gray, None)
        if des is None or len(kp) == 0:
            logging.warning(f"No SIFT features found for {self.current_image_path}, cannot add to reference store.")
            return

        rect = self.current_rect_item.rect()
        bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
        class_id = self.class_combo.currentIndex()
        
        # Remove old entry from in-memory cache if it exists
        self.references = [ref for ref in self.references if ref['image_path'] != self.current_image_path]
        
        # Add new entry to in-memory cache
        new_ref = {'image_path': self.current_image_path, 'kp': kp, 'des': des, 'bbox': bbox, 'class_id': class_id}
        self.references.append(new_ref)
        
        # Now, save to the HDF5 file
        try:
            with h5py.File(self.h5_ref_path, 'a') as hf:
                group_name = _sanitize_path_for_h5(os.path.basename(self.current_image_path))
                if group_name in hf:
                    del hf[group_name] # Delete old entry to ensure update
                
                group = hf.create_group(group_name)
                group.attrs['image_path'] = os.path.basename(self.current_image_path)
                group.create_dataset('keypoints', data=_keypoints_to_array(kp))
                group.create_dataset('descriptors', data=des)
                group.create_dataset('bbox', data=np.array(bbox))
                group.create_dataset('class_id', data=class_id)
        except Exception as e:
            logging.error(f"Failed to write to HDF5 store: {e}")
            QMessageBox.critical(self, "HDF5 Write Error", f"Could not save reference to references.h5:\n{e}")

        self.manual_annotation_count = len(self.references)
        self.update_auto_annotate_button_status()
        self.statusBar.showMessage(f"Manual reference updated. Live pool size: {self.manual_annotation_count}", 3000)
        logging.info(f"Updated HDF5 store for {self.current_image_path}. Total refs: {self.manual_annotation_count}")

    # --- MODIFIED FOR HDF5 ---
    def clear_annotation(self):
        if self.current_rect_item:
            self.scene.removeItem(self.current_rect_item)
            self.current_rect_item = None
        
        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        if os.path.exists(txt_path):
            try:
                os.remove(txt_path)
                logging.info(f"Deleted annotation file: {txt_path}")
                self.update_list_item_status(self.current_index)
            except OSError as e:
                logging.error(f"Failed to delete {txt_path}: {e}")
        
        # Remove from HDF5 store if it was a reference
        group_name = _sanitize_path_for_h5(os.path.basename(self.current_image_path))
        is_in_cache = any(ref['image_path'] == self.current_image_path for ref in self.references)

        if is_in_cache:
            # Remove from live cache
            self.references = [ref for ref in self.references if ref['image_path'] != self.current_image_path]
            
            # Remove from HDF5 file
            if self.h5_ref_path and os.path.exists(self.h5_ref_path):
                try:
                    with h5py.File(self.h5_ref_path, 'a') as hf:
                        if group_name in hf:
                            del hf[group_name]
                            logging.info(f"Removed '{group_name}' from HDF5 store.")
                except Exception as e:
                    logging.error(f"Failed to update HDF5 on clear: {e}")
            
            self.manual_annotation_count = len(self.references)
            self.update_auto_annotate_button_status()
            self.statusBar.showMessage(f"Removed manual reference. Pool size: {self.manual_annotation_count}", 3000)
            
    # --- MODIFIED FOR HDF5 ---
    def delete_current_image(self):
        if self.current_index == -1: return

        img_path = self.current_image_path
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        reply = QMessageBox.question(self, 'Confirm Deletion', f"Permanently delete {os.path.basename(img_path)}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Yes:
            # First, remove it from references if it is one
            self.clear_annotation()

            # Then delete the files
            try:
                os.remove(img_path)
                if os.path.exists(txt_path): os.remove(txt_path)
            except OSError as e:
                QMessageBox.warning(self, "Delete Error", f"Error deleting files:\n{e}")
                return

            deleted_index = self.current_index
            self.image_list.pop(deleted_index)
            self.image_list_widget.takeItem(deleted_index)
            
            if not self.image_list: self.load_image(-1)
            else: self.load_image(min(deleted_index, len(self.image_list) - 1))
    
    # --- No changes needed in the functions below ---
    def next_image(self):
        if self.current_rect_item: self.save_annotation(silent=True)
        if self.current_index < len(self.image_list) - 1: self.load_image(self.current_index + 1)

    def prev_image(self):
        if self.current_rect_item: self.save_annotation(silent=True)
        if self.current_index > 0: self.load_image(self.current_index - 1)

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
            self.pixmap_item, self.current_rect_item, self.current_index = None, None, -1
            self.statusBar.showMessage("No images to display.")
            return

        if self.pixmap_item: self.scene.removeItem(self.pixmap_item)
        if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
        self.pixmap_item, self.current_rect_item = None, None
        
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

        # Suggest auto-annotation if enough references exist and image is blank
        if self.manual_annotation_count >= 10 and not self.current_rect_item:
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
            if not content: item.setBackground(QBrush(QColor("#FFA500"))) # Orange for empty
            else: item.setBackground(QBrush(QColor("#90EE90"))) # Green for annotated
        except Exception as e:
            item.setBackground(QBrush(QColor("transparent")))
            logging.error(f"Could not read status for {txt_path}: {e}")
            
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

    def update_auto_annotate_button_status(self):
        needed = 10
        if self.manual_annotation_count >= needed:
            self.auto_btn.setEnabled(True)
            self.auto_btn.setText("Auto-Annotate (Ready)")
            if self.manual_annotation_count == needed:
                logging.info("Auto-annotate feature enabled.")
        else:
            self.auto_btn.setEnabled(False)
            self.auto_btn.setText(f"Auto-Annotate ({self.manual_annotation_count}/{needed})")
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnnotationApp()
    window.show()
    sys.exit(app.exec())