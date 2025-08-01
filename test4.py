import sys
import os
import glob
import cv2
import numpy as np
import logging
import h5py
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QGraphicsView, QGraphicsScene, QFileDialog,
    QMessageBox, QStatusBar, QGraphicsRectItem, QGraphicsPixmapItem,
    QListWidget, QListWidgetItem, QCompleter, QLabel
)
from PyQt6.QtGui import QPixmap, QPen, QPainter, QColor, QBrush, QCursor
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QStringListModel, QObject, QThread

# --- Setup Logging ---
logging.basicConfig(
    filename='annotation_tool.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- HDF5 LOGIC ---
def _keypoints_to_array(kp):
    return np.array([[p.pt[0], p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id] for p in kp])

def _array_to_keypoints(arr):
    return [cv2.KeyPoint(x=p[0], y=p[1], _size=p[2], _angle=p[3], _response=p[4], _octave=int(p[5]), _class_id=int(p[6])) for p in arr]

def _sanitize_path_for_h5(path):
    return path.replace('/', '_').replace('\\', '_').replace('.', '_')

# --- THREADING: WORKER CLASS ---
class Worker(QObject):
    """
    Worker object to handle long-running tasks in a separate thread.
    """
    reference_saved = pyqtSignal(str)
    reference_deleted = pyqtSignal(str)
    features_prefetched = pyqtSignal(str, object, object, object)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.sift = cv2.SIFT_create(nfeatures=2000)

    def do_save_reference(self, h5_path, image_path, rect_data, class_id, kp_arr, des):
        """Saves a new reference to the HDF5 file using pre-computed features."""
        try:
            logging.info(f"[Worker] Starting reference save for {image_path}")
            if des is None or len(kp_arr) == 0:
                logging.warning(f"[Worker] No SIFT features provided for {image_path}, cannot add to store.")
                return

            bbox = [rect_data.left(), rect_data.top(), rect_data.right(), rect_data.bottom()]
            group_name = _sanitize_path_for_h5(os.path.basename(image_path))

            with h5py.File(h5_path, 'a') as hf:
                if group_name in hf:
                    del hf[group_name]
                group = hf.create_group(group_name)
                group.attrs['image_path'] = os.path.basename(image_path)
                group.create_dataset('keypoints', data=kp_arr)
                group.create_dataset('descriptors', data=des)
                group.create_dataset('bbox', data=np.array(bbox))
                group.create_dataset('class_id', data=class_id)

            logging.info(f"[Worker] Successfully saved reference for {image_path} to HDF5.")
            self.reference_saved.emit(image_path)
        except Exception as e:
            self.error.emit(f"Error saving reference:\n{traceback.format_exc()}")

    def do_delete_reference(self, h5_path, image_path):
        """Deletes a reference from the HDF5 file."""
        try:
            if not h5_path or not os.path.exists(h5_path):
                return
            group_name = _sanitize_path_for_h5(os.path.basename(image_path))
            logging.info(f"[Worker] Deleting reference {group_name} from {h5_path}")
            with h5py.File(h5_path, 'a') as hf:
                if group_name in hf:
                    del hf[group_name]
                    logging.info(f"[Worker] Removed '{group_name}' from HDF5 store.")
                    self.reference_deleted.emit(image_path)
        except Exception as e:
            self.error.emit(f"Error deleting reference from HDF5:\n{traceback.format_exc()}")

    def do_prefetch_image(self, image_path):
        """Loads an image and computes its features in advance."""
        try:
            if not os.path.exists(image_path): return
            logging.info(f"[Worker] Prefetching {image_path}")
            pixmap = QPixmap(image_path)
            if pixmap.isNull(): return
            img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None: return
            kp, des = self.sift.detectAndCompute(img_gray, None)
            self.features_prefetched.emit(image_path, pixmap, kp, des)
        except Exception as e:
            self.error.emit(f"Error prefetching image:\n{traceback.format_exc()}")


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


class AnnotationApp(QMainWindow):
    trigger_save_reference = pyqtSignal(str, str, QRectF, int, np.ndarray, np.ndarray)
    trigger_prefetch = pyqtSignal(str)
    trigger_delete_reference = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuickDraw - Smart Reference Annotator (HDF5 Edition)")
        self.setGeometry(100, 100, 1200, 800)

        # SIFT/FLANN for the main thread (used for auto-annotate)
        self.sift = cv2.SIFT_create(nfeatures=2000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # State attributes
        self.references = []
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
        self.h5_ref_path = None
        self.current_annotation_is_manual = False
        self.annotation_is_dirty = False  # Flag to track if the current annotation needs saving

        # Threading/caching attributes
        self.prefetched_data = None
        self.current_image_features = (None, None) # (kp, des) for current image

        self.init_ui()
        self.init_worker_thread()
        logging.info("Application initialized with HDF5, SIFT/FLANN and Worker Thread.")

    def init_ui(self):
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

    def init_worker_thread(self):
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.worker.reference_saved.connect(self.on_reference_saved)
        self.worker.reference_deleted.connect(self.on_reference_deleted)
        self.worker.features_prefetched.connect(self.on_features_prefetched)
        self.worker.error.connect(self.on_worker_error)
        self.trigger_save_reference.connect(self.worker.do_save_reference)
        self.trigger_delete_reference.connect(self.worker.do_delete_reference)
        self.trigger_prefetch.connect(self.worker.do_prefetch_image)
        self.thread.start()

    def on_reference_saved(self, image_path):
        self.statusBar.showMessage(f"Background save complete for {os.path.basename(image_path)}.", 3000)

    def on_reference_deleted(self, image_path):
        self.statusBar.showMessage(f"Removed reference {os.path.basename(image_path)} from HDF5 store.", 3000)
        self.manual_annotation_count = len(self.references)
        self.update_auto_annotate_button_status()

    def on_features_prefetched(self, path, pixmap, kp, des):
        logging.info(f"Received prefetched data for {os.path.basename(path)}")
        self.prefetched_data = {'path': path, 'pixmap': pixmap, 'kp': kp, 'des': des}

    def on_worker_error(self, error_message):
        logging.error(f"Error from worker thread: {error_message}")
        QMessageBox.critical(self, "Worker Thread Error", error_message)

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
        self.h5_ref_path = os.path.join(folder, 'references.h5')
        self.load_references_from_h5()
        self.populate_image_list_panel()
        if self.image_list:
            self.current_index = 0
            self.load_image(self.current_index)

    def auto_annotate(self, silent=False):
        kp1, des1 = self.current_image_features
        if des1 is None or len(kp1) == 0:
            if not silent: QMessageBox.warning(self, "Auto-Annotate", "Could not find SIFT features in the current image.")
            return

        best_match = {'ref': None, 'score': -1, 'M': None}
        MIN_MATCH_COUNT = 10
        target_class_id = self.class_combo.currentIndex()
        relevant_references = [ref for ref in self.references if ref['class_id'] == target_class_id] if target_class_id >= 0 else self.references

        for ref in relevant_references:
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

        if best_match['ref'] is None:
            if not silent: QMessageBox.information(self, "Auto-Annotate", f"Failed to find a suitable match. Best score: {best_match['score']}")
            return

        ref, M = best_match['ref'], best_match['M']
        ref_bbox = ref['bbox']
        ref_pts = np.float32([[ref_bbox[0], ref_bbox[1]], [ref_bbox[0], ref_bbox[3]], [ref_bbox[2], ref_bbox[3]], [ref_bbox[2], ref_bbox[1]]]).reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(ref_pts, M)
        x_min, y_min = np.min(transformed_pts, axis=0)[0]
        x_max, y_max = np.max(transformed_pts, axis=0)[0]
        img_w, img_h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(img_w, x_max), min(img_h, y_max)
        new_rect = QRectF(QPointF(x_min, y_min), QPointF(x_max, y_max))
        self.draw_bbox(new_rect, Qt.GlobalColor.blue)
        if ref['class_id'] < self.class_combo.count(): self.class_combo.setCurrentIndex(ref['class_id'])

        self.current_annotation_is_manual = False
        self.annotation_is_dirty = True  # Mark that this annotation needs saving
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

        if self.current_annotation_is_manual:
            self.statusBar.showMessage("Submitting reference to background saver...", 2000)
            kp, des = self.current_image_features
            if des is not None:
                kp_arr = _keypoints_to_array(kp)
                self.trigger_save_reference.emit(self.h5_ref_path, self.current_image_path, rect, class_id, kp_arr, des)
                self.update_in_memory_reference(rect, class_id)

        self.current_annotation_is_manual = False
        self.annotation_is_dirty = False  # Reset dirty flag after saving
        self.update_list_item_status(self.current_index)
        self.draw_bbox(rect, Qt.GlobalColor.red)
        if not silent:
            self.statusBar.showMessage(f"Annotation saved to {os.path.basename(txt_path)}", 3000)
        logging.info(f"Annotation saved for {self.current_image_path}")

    def update_in_memory_reference(self, rect, class_id):
        kp, des = self.current_image_features
        if des is None: return
        bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
        self.references = [ref for ref in self.references if ref['image_path'] != self.current_image_path]
        new_ref = {'image_path': self.current_image_path, 'kp': kp, 'des': des, 'bbox': bbox, 'class_id': class_id}
        self.references.append(new_ref)
        self.manual_annotation_count = len(self.references)
        self.update_auto_annotate_button_status()

    def load_image(self, index):
        if not (0 <= index < len(self.image_list)):
            self.current_index = -1
            self.current_image_path = None
            if self.pixmap_item: self.scene.removeItem(self.pixmap_item)
            if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
            self.pixmap_item, self.current_rect_item = None, None
            self.scene.clear()
            self.statusBar.showMessage("No image selected.")
            return

        target_path = self.image_list[index]
        pixmap, kp, des = None, None, None
        if self.prefetched_data and self.prefetched_data['path'] == target_path:
            logging.info(f"Using prefetched data for {os.path.basename(target_path)}")
            pixmap, kp, des = self.prefetched_data['pixmap'], self.prefetched_data['kp'], self.prefetched_data['des']
            self.prefetched_data = None
        else:
            logging.info(f"Loading {os.path.basename(target_path)} from disk.")
            pixmap = QPixmap(target_path)
            if not pixmap.isNull():
                img_gray = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is not None:
                    kp, des = self.sift.detectAndCompute(img_gray, None)

        if pixmap.isNull():
            QMessageBox.warning(self, "Load Error", f"Failed to load image: {target_path}")
            return

        if self.pixmap_item: self.scene.removeItem(self.pixmap_item)
        if self.current_rect_item: self.scene.removeItem(self.current_rect_item)
        self.pixmap_item, self.current_rect_item = None, None

        self.current_index = index
        self.image_list_widget.setCurrentRow(index)
        self.current_image_path = target_path
        self.current_image_features = (kp, des)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(-1)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.statusBar.showMessage(f"Image {index + 1}/{len(self.image_list)}: {os.path.basename(self.current_image_path)}")
        self.load_existing_annotation()

        if self.manual_annotation_count >= 10 and not self.current_rect_item:
            self.auto_annotate(silent=True)
        else:
            self.annotation_is_dirty = False # Reset flag if no auto-annotation was made

        next_index = index + 1
        if next_index < len(self.image_list):
            self.trigger_prefetch.emit(self.image_list[next_index])

    def closeEvent(self, event):
        logging.info("Shutting down worker thread.")
        self.thread.quit()
        if not self.thread.wait(3000):
            logging.warning("Worker thread did not shut down cleanly. Terminating.")
            self.thread.terminate()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_W: self.toggle_annotation_mode()
        elif key == Qt.Key.Key_A: self.prev_image()
        elif key == Qt.Key.Key_D: self.next_image()
        elif key == Qt.Key.Key_Escape: self.clear_annotation()
        else: super().keyPressEvent(event)

    def next_image(self):
        if self.annotation_is_dirty:
            self.save_annotation(silent=True)
        if self.current_index < len(self.image_list) - 1:
            self.load_image(self.current_index + 1)

    def prev_image(self):
        if self.annotation_is_dirty:
            self.save_annotation(silent=True)
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def on_list_item_select(self, current_item, previous_item):
        if current_item is None: return
        row = self.image_list_widget.row(current_item)
        if row != self.current_index:
            if previous_item is not None and self.annotation_is_dirty:
                self.save_annotation(silent=True)
            self.load_image(row)

    def clear_annotation(self):
        if self.current_rect_item:
            self.scene.removeItem(self.current_rect_item)
            self.current_rect_item = None
            self.annotation_is_dirty = False # Nothing to save, clear the flag

        txt_path = os.path.splitext(self.current_image_path)[0] + '.txt'
        if os.path.exists(txt_path):
            try:
                os.remove(txt_path)
                logging.info(f"Deleted annotation file: {txt_path}")
                self.update_list_item_status(self.current_index)
            except OSError as e:
                logging.error(f"Failed to delete {txt_path}: {e}")

        is_in_cache = any(ref['image_path'] == self.current_image_path for ref in self.references)
        if is_in_cache:
            self.references = [ref for ref in self.references if ref['image_path'] != self.current_image_path]
            self.statusBar.showMessage(f"Removing reference...", 3000)
            self.trigger_delete_reference.emit(self.h5_ref_path, self.current_image_path)

    def load_references_from_h5(self):
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
                    img_path = os.path.join(image_folder, group.attrs['image_path'])
                    if not os.path.exists(img_path):
                        logging.warning(f"Reference image {img_path} not found. Skipping.")
                        continue
                    kp_arr = group['keypoints'][:]
                    des = group['descriptors'][:]
                    bbox = group['bbox'][:]
                    class_id = group['class_id'][()]
                    ref = {'image_path': img_path, 'kp': _array_to_keypoints(kp_arr), 'des': des, 'bbox': list(bbox), 'class_id': class_id}
                    self.references.append(ref)
        except Exception as e:
            logging.error(f"Failed to load HDF5 reference file: {e}")
            QMessageBox.critical(self, "HDF5 Load Error", f"Could not read references.h5:\n{e}")
        self.manual_annotation_count = len(self.references)
        self.update_auto_annotate_button_status()
        self.statusBar.showMessage(f"Loaded {len(self.references)} references from HDF5.", 4000)

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
        self.current_annotation_is_manual = True
        self.annotation_is_dirty = True # Mark that this manual annotation needs saving

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

    def delete_current_image(self):
        if self.current_index == -1: return
        img_path = self.current_image_path
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        reply = QMessageBox.question(self, 'Confirm Deletion', f"Permanently delete {os.path.basename(img_path)}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Yes:
            self.clear_annotation()
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnnotationApp()
    window.show()
    sys.exit(app.exec())