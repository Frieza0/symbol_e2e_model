import sys
import os
import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel, QFileDialog,
    QMessageBox, QHeaderView, QAbstractItemView, QRubberBand, QComboBox, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QRect, QPoint, QSize


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


class PinImageLabel(QLabel):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(1500, 1000)
        self.pixmap = None
        self.pins = []
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.current_mode = "pin"  # "pin" or "name"

    def set_image(self, image_path):
        self.pixmap = QPixmap(image_path)
        self.update()

    def set_pins(self, pins):
        self.pins = pins
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw scaled image
        scaled = self.pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x_offset, y_offset, scaled)

        if self.pixmap.width() == 0 or self.pixmap.height() == 0:
            return

        scale_x = scaled.width() / self.pixmap.width()
        scale_y = scaled.height() / self.pixmap.height()

        pen = QPen(QColor("red"), 2)
        painter.setPen(pen)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        for pin in self.pins:
            # === 1. 绘制 Pin# 框 ===
            if "bbox_pin" in pin and pin["bbox_pin"] is not None:
                x1, y1, x2, y2 = pin["bbox_pin"]
                px1 = x_offset + int(x1 * scale_x)
                py1 = y_offset + int(y1 * scale_y)
                px2 = x_offset + int(x2 * scale_x)
                py2 = y_offset + int(y2 * scale_y)
                painter.drawRect(px1, py1, px2 - px1, py2 - py1)
                painter.drawText(px1 + 5, py1 + 15, str(pin["pin_number"]))

            # === 2. 绘制 Name 框 ===
            if "bbox_name" in pin and pin["bbox_name"] is not None:
                x1, y1, x2, y2 = pin["bbox_name"]
                px1 = x_offset + int(x1 * scale_x)
                py1 = y_offset + int(y1 * scale_y)
                px2 = x_offset + int(x2 * scale_x)
                py2 = y_offset + int(y2 * scale_y)
                painter.drawRect(px1, py1, px2 - px1, py2 - py1)
                painter.drawText(px1 + 5, py1 + 15, pin["name"])

    def mousePressEvent(self, event):
        if not self.pixmap:
            return
        self.origin = event.pos()
        self.rubberBand.setGeometry(QRect(self.origin, QSize()))
        self.rubberBand.show()

    def mouseMoveEvent(self, event):
        if self.rubberBand.isVisible():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if not self.pixmap or not self.rubberBand.isVisible():
            return
        self.rubberBand.hide()

        selected_row = self.main_window.get_selected_row()
        if selected_row is None:
            QMessageBox.warning(self, "提示", "请先在表格中选中一个引脚行再框选区域。")
            return

        rect = self.rubberBand.geometry()
        scaled = self.pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2

        if not (x_offset <= rect.left() and rect.right() <= x_offset + scaled.width() and
                y_offset <= rect.top() and rect.bottom() <= y_offset + scaled.height()):
            QMessageBox.warning(self, "警告", "框选区域必须完全在图像内。")
            return

        left_lbl = rect.left()
        top_lbl = rect.top()
        right_lbl = rect.right()
        bottom_lbl = rect.bottom()

        x1_orig = int((left_lbl - x_offset) / (scaled.width() / self.pixmap.width()))
        y1_orig = int((top_lbl - y_offset) / (scaled.height() / self.pixmap.height()))
        x2_orig = int((right_lbl - x_offset) / (scaled.width() / self.pixmap.width()))
        y2_orig = int((bottom_lbl - y_offset) / (scaled.height() / self.pixmap.height()))

        x1_orig, x2_orig = sorted([x1_orig, x2_orig])
        y1_orig, y2_orig = sorted([y1_orig, y2_orig])

        if self.current_mode == "pin":
            self.main_window.update_pin_bbox(selected_row, "pin", x1_orig, y1_orig, x2_orig, y2_orig)
        elif self.current_mode == "name":
            self.main_window.update_pin_bbox(selected_row, "name", x1_orig, y1_orig, x2_orig, y2_orig)

        # 👇 关键：刷新 pins 并更新显示
        self.main_window.refresh_pins_from_table()
        self.set_pins(self.main_window.pins)

        # ✅ 自动跳到下一行（核心新增代码）
        total_rows = self.main_window.table.rowCount()
        next_row = selected_row + 1
        if next_row < total_rows:
            self.main_window.table.selectRow(next_row)
            self.main_window.table.scrollToItem(
                self.main_window.table.item(next_row, 0),
                QAbstractItemView.ScrollHint.PositionAtCenter
            )
        else:
            # 可选：提示已完成
            # QMessageBox.information(self, "完成", "所有引脚已标注完毕！")
            pass


class PinAnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pin Annotator - 双框标注模式")
        self.resize(1200, 800)
        
        self.dataset_dir = None
        self.image_files = []
        self.current_index = 0
        self.current_image_path = None
        self.pins = []
        self.package_type = "2side"
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.image_label = PinImageLabel(main_window=self)
        layout.addWidget(self.image_label, 4)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        layout.addWidget(right_widget, 3)

        # Table: 4 columns
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Pin#", "Box (Pin#)", "Name", "Box (Name)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        right_layout.addWidget(self.table)

        # Mode selector
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["框选 Pin#", "框选 Name"])
        self.mode_combo.currentIndexChanged.connect(
            lambda idx: setattr(self.image_label, 'current_mode', self.mode_combo.itemText(idx).split()[1].lower())
        )
        right_layout.addWidget(self.mode_combo)

        # Package Type Selector
        self.package_combo = QComboBox()
        self.package_combo.addItems(["1side", "2side", "4side", "circle", "grid"])
        self.package_combo.setCurrentText(self.package_type)
        self.package_combo.currentTextChanged.connect(self.on_package_type_changed)
        right_layout.addWidget(QLabel("封装类型:"))
        right_layout.addWidget(self.package_combo)

        # 👇 新增：跳转控件
        jump_layout = QHBoxLayout()
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("输入图像序号 (1～N)")
        self.jump_input.setFixedWidth(120)
        self.jump_btn = QPushButton("跳转")
        self.jump_btn.clicked.connect(self.jump_to_image)
        self.status_label = QLabel("共 0 张")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        jump_layout.addWidget(QLabel("跳转到:"))
        jump_layout.addWidget(self.jump_input)
        jump_layout.addWidget(self.jump_btn)
        jump_layout.addWidget(self.status_label)
        right_layout.addLayout(jump_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("➕ 添加")
        self.del_btn = QPushButton("🗑️ 删除")
        self.save_btn = QPushButton("💾 保存")
        self.prev_btn = QPushButton("◀ 上一张")
        self.next_btn = QPushButton("下一张 ▶")

        self.add_btn.clicked.connect(self.add_pin_row)
        self.del_btn.clicked.connect(self.delete_selected_row)
        self.save_btn.clicked.connect(self.save_annotation)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)

        for btn in [self.prev_btn, self.next_btn, self.add_btn, self.del_btn, self.save_btn]:
            btn_layout.addWidget(btn)
        right_layout.addLayout(btn_layout)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        open_action = file_menu.addAction("📁 打开数据集目录")
        open_action.triggered.connect(self.open_dataset)

        # 初始化状态
        self.update_status_label()


    def update_status_label(self):
        total = len(self.image_files)
        current = self.current_index + 1 if total > 0 else 0
        self.status_label.setText(f"第 {current} / {total} 张")


    def jump_to_image(self):
        if not self.image_files:
            QMessageBox.warning(self, "提示", "请先打开数据集目录！")
            return

        text = self.jump_input.text().strip()
        if not text.isdigit():
            QMessageBox.warning(self, "输入错误", "请输入有效的数字！")
            return

        target = int(text)
        total = len(self.image_files)

        if target < 1 or target > total:
            QMessageBox.warning(self, "范围错误", f"请输入 1 到 {total} 之间的数字！")
            return

        self.current_index = target - 1  # 转为 0-based
        self.load_current_image()
        self.update_status_label()
        self.jump_input.clear()  # 清空输入框

    def on_package_type_changed(self, text):
        self.package_type = text

    def get_selected_row(self):
        selected = self.table.selectedItems()
        if not selected:
            return None
        rows = {item.row() for item in selected}
        return list(rows)[0] if len(rows) == 1 else None

    def update_pin_bbox(self, row, key, x1, y1, x2, y2):
        if key == "pin":
            col = 1
            item = self.table.item(row, col)
            if item:
                item.setText(f"{x1},{y1},{x2},{y2}")
        elif key == "name":
            col = 3
            item = self.table.item(row, col)
            if item:
                item.setText(f"{x1},{y1},{x2},{y2}")

    def refresh_pins_from_table(self):
        self.pins = self.get_pins_from_table_silent()

    def get_pins_from_table_silent(self):
        pins = []
        for row in range(self.table.rowCount()):
            try:
                pin_num = int(self.table.item(row, 0).text())
                name = self.table.item(row, 2).text().strip()
                bbox_pin_text = self.table.item(row, 1).text()
                bbox_name_text = self.table.item(row, 3).text()

                bbox_pin = None
                if bbox_pin_text:
                    coords = list(map(int, bbox_pin_text.split(',')))
                    if len(coords) == 4:
                        bbox_pin = coords

                bbox_name = None
                if bbox_name_text:
                    coords = list(map(int, bbox_name_text.split(',')))
                    if len(coords) == 4:
                        bbox_name = coords

                pins.append({
                    "pin_number": pin_num,
                    "name": name,
                    "bbox_pin": bbox_pin,
                    "bbox_name": bbox_name
                })
            except Exception:
                continue
        return pins

    def open_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集目录（需含 images/ 和 annotations/）")
        if not folder:
            return
        img_dir = Path(folder) / "images"
        anno_dir = Path(folder) / "annotations"
        if not (img_dir.exists() and anno_dir.exists()):
            QMessageBox.critical(self, "错误", "目录必须包含 images/ 和 annotations/ 子文件夹！")
            return

        self.dataset_dir = Path(folder)
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.image_files = sorted([
            f.name for f in img_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not self.image_files:
            QMessageBox.warning(self, "警告", "images/ 目录中无图像文件！")
            return

        self.current_index = 0
        self.load_current_image()

    def load_current_image(self):
        if not self.image_files:
            return
        img_name = self.image_files[self.current_index]
        self.current_image_path = self.img_dir / img_name
        self.image_label.set_image(str(self.current_image_path))
        self.setWindowTitle(f"Pin Annotator - {img_name} (第 {self.current_index+1}/{len(self.image_files)} 张)")

        anno_path = self.anno_dir / (Path(img_name).stem + ".json")
        self.pins = []
        if anno_path.exists():
            try:
                with open(anno_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.pins = data.get("pins", [])
                self.package_type = data.get("package_type", "2side")  # 👈 恢复
                self.package_combo.setCurrentText(self.package_type)   # 👈 同步 UI
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"JSON 错误: {e}")
        else:
            # 新图像：重置为默认
            self.package_type = "2side"
            self.package_combo.setCurrentText(self.package_type)

        self.refresh_table()
        self.image_label.set_pins(self.pins)

    def refresh_table(self):
        # 先按 pin_number 排序
        from natsort import natsorted

        sorted_pins = natsorted(self.pins, key=lambda x: x["pin_number"])

        self.table.setRowCount(len(sorted_pins))
        for row, pin in enumerate(sorted_pins):
            num_item = QTableWidgetItem(str(pin["pin_number"]))
            name_item = QTableWidgetItem(pin.get("name", ""))
            bbox_pin = pin.get("bbox_pin")
            bbox_name = pin.get("bbox_name")

            bbox_pin_item = QTableWidgetItem("")
            bbox_name_item = QTableWidgetItem("")

            if bbox_pin:
                bbox_pin_item.setText(f"{bbox_pin[0]},{bbox_pin[1]},{bbox_pin[2]},{bbox_pin[3]}")
            if bbox_name:
                bbox_name_item.setText(f"{bbox_name[0]},{bbox_name[1]},{bbox_name[2]},{bbox_name[3]}")

            for item in (num_item, name_item, bbox_pin_item, bbox_name_item):
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(row, 0, num_item)
            self.table.setItem(row, 1, bbox_pin_item)
            self.table.setItem(row, 2, name_item)
            self.table.setItem(row, 3, bbox_name_item)

    def add_pin_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("1"))
        self.table.setItem(row, 1, QTableWidgetItem(""))  # Box (Pin#)
        self.table.setItem(row, 2, QTableWidgetItem(""))  # Name
        self.table.setItem(row, 3, QTableWidgetItem(""))  # Box (Name)

    def delete_selected_row(self):
        rows = set(item.row() for item in self.table.selectedItems())
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def save_annotation(self):
        pins = self.get_pins_from_table()
        if pins is None:
            return

        img_path = Path(self.current_image_path)
        data = {
            "image_id": img_path.stem,
            "width": QImage(str(img_path)).width(),
            "height": QImage(str(img_path)).height(),
            "total_pins": len(pins),
            "package_type": self.package_type,  # 👈 新增字段
            "pins": pins
        }

        anno_path = self.anno_dir / (img_path.stem + ".json")
        try:
            with open(anno_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # QMessageBox.information(self, "成功", "标注已保存！")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def get_pins_from_table(self):
        pins = []
        for row in range(self.table.rowCount()):
            try:
                pin_num = int(self.table.item(row, 0).text())
                name = self.table.item(row, 2).text().strip()
                bbox_pin_text = self.table.item(row, 1).text()
                bbox_name_text = self.table.item(row, 3).text()

                bbox_pin = None
                if bbox_pin_text:
                    coords = list(map(int, bbox_pin_text.split(',')))
                    if len(coords) != 4:
                        raise ValueError("Pin# 框需要4个整数")
                    bbox_pin = coords

                bbox_name = None
                if bbox_name_text:
                    coords = list(map(int, bbox_name_text.split(',')))
                    if len(coords) != 4:
                        raise ValueError("Name 框需要4个整数")
                    bbox_name = coords

                pins.append({
                    "pin_number": pin_num,
                    "name": name,
                    "bbox_pin": bbox_pin,
                    "bbox_name": bbox_name
                })
            except Exception as e:
                QMessageBox.warning(self, "格式错误", f"第 {row+1} 行错误: {e}")
                return None
        return pins

    def next_image(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.load_current_image()

    def prev_image(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.load_current_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PinAnnotatorApp()
    window.show()
    sys.exit(app.exec())