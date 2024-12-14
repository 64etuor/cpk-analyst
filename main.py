# main.py

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QProgressBar,
    QLineEdit, QVBoxLayout, QHBoxLayout, QMessageBox, QTableWidget,
    QTableWidgetItem, QSpinBox, QComboBox, QCheckBox, QHeaderView,
    QAbstractItemView
)
from PyQt5.QtGui import QClipboard
from PyQt5.QtCore import Qt
from scipy.special import gamma as gamma_function


class CopyableQTableWidget(QTableWidget):
    def keyPressEvent(self, event):        
        if event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            self.copy_selection()
        else:
            super().keyPressEvent(event)

    def copy_selection(self):
        selection = self.selectedIndexes()
        if not selection:
            return
        
        row_min = min(index.row() for index in selection)
        row_max = max(index.row() for index in selection)
        col_min = min(index.column() for index in selection)
        col_max = max(index.column() for index in selection)

        data_str = ""
        for row in range(row_min, row_max + 1):
            row_data = []
            for col in range(col_min, col_max + 1):
                item = self.item(row, col)
                row_data.append(item.text() if item else "")
            data_str += "\t".join(row_data) + "\n"

        clipboard = QApplication.clipboard()
        clipboard.setText(data_str.strip())


class ProcessCapabilityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Process Capability Analysis Program'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(800, 650)

        # Main layout
        main_layout = QVBoxLayout()

        # Create UI components
        file_layout = self.create_file_selection_layout()
        sheet_layout = self.create_sheet_selection_layout()
        progress_and_status_layout = self.create_progress_and_status_widgets()
        data_table_widget = self.create_data_table()
        input_layout = self.create_input_layout()
        result_table_widget = self.create_result_table()

        # Add components to main layout
        main_layout.addLayout(file_layout)
        main_layout.addLayout(sheet_layout)
        main_layout.addLayout(progress_and_status_layout)
        main_layout.addWidget(data_table_widget)
        main_layout.addLayout(input_layout)
        main_layout.addWidget(result_table_widget)

        self.setLayout(main_layout)

    def create_file_selection_layout(self):
        file_layout = QHBoxLayout()
        self.file_label = QLabel('Select a file (Excel or CSV):')
        self.file_path = QLineEdit()
        self.file_path.setMinimumWidth(400)
        self.file_path.setMaximumWidth(800)
        self.browse_button = QPushButton('Browse')
        self.browse_button.setMaximumWidth(200)
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(self.browse_button)
        return file_layout

    def create_sheet_selection_layout(self):
        sheet_layout = QHBoxLayout()
        self.sheet_label = QLabel('Select a sheet:')
        self.sheet_combo = QComboBox()
        self.sheet_combo.setMaximumWidth(200)
        self.sheet_combo.currentIndexChanged.connect(self.load_sheet_data)
        sheet_layout.addWidget(self.sheet_label)
        sheet_layout.addWidget(self.sheet_combo)
        return sheet_layout

    def create_progress_and_status_widgets(self):
        progress_and_status_layout = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setVisible(False)
        self.status_label = QLabel('Status: Ready')
        progress_and_status_layout.addWidget(self.progress_bar)
        progress_and_status_layout.addWidget(self.status_label)
        progress_and_status_layout.addStretch()
        return progress_and_status_layout

    def create_data_table(self):
        self.data_table = QTableWidget()
        self.data_table.setSelectionBehavior(QTableWidget.SelectColumns)
        self.data_table.setSelectionMode(QTableWidget.MultiSelection)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.data_table.horizontalHeader().setStretchLastSection(False)
        self.data_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.data_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.data_table.setSizeAdjustPolicy(QAbstractItemView.AdjustToContents)
        return self.data_table

    def create_spec_limit_inputs(self):
        spec_layout = QHBoxLayout()
        self.lsl_label = QLabel('LSL (Lower Spec Limit):')
        self.lsl_input = QLineEdit()
        self.lsl_input.setMaximumWidth(100)
        self.usl_label = QLabel('USL (Upper Spec Limit):')
        self.usl_input = QLineEdit()
        self.usl_input.setMaximumWidth(100)
        spec_layout.addWidget(self.lsl_label)
        spec_layout.addWidget(self.lsl_input)
        spec_layout.addWidget(self.usl_label)
        spec_layout.addWidget(self.usl_input)
        spec_layout.addStretch()
        return spec_layout

    def create_subgroup_size_input(self):
        subgroup_layout = QHBoxLayout()
        self.subgroup_label = QLabel('Subgroup Size:')
        self.subgroup_input = QSpinBox()
        self.subgroup_input.setValue(1)
        self.subgroup_input.setMinimum(1)
        self.subgroup_input.setMaximumWidth(100)
        subgroup_layout.addWidget(self.subgroup_label)
        subgroup_layout.addWidget(self.subgroup_input)
        subgroup_layout.addStretch()
        return subgroup_layout

    def create_bins_input(self):
        bins_layout = QHBoxLayout()
        self.bins_label = QLabel('Number of bins:')
        self.bins_input = QSpinBox()
        self.bins_input.setValue(15)
        self.bins_input.setMinimum(1)
        self.bins_input.setMaximum(100)
        self.bins_input.setMaximumWidth(100)
        bins_layout.addWidget(self.bins_label)
        bins_layout.addWidget(self.bins_input)
        bins_layout.addStretch()
        return bins_layout

    def create_options_checkboxes(self):
        options_layout = QHBoxLayout()
        self.remove_zeros_checkbox = QCheckBox('Remove zero values')
        self.remove_negatives_checkbox = QCheckBox('Remove negative values')
        options_layout.addWidget(self.remove_zeros_checkbox)
        options_layout.addWidget(self.remove_negatives_checkbox)
        options_layout.addStretch()
        return options_layout

    def create_analyze_button_and_info_label(self):
        buttons_layout = QHBoxLayout()
        self.analyze_button = QPushButton('Start Analysis')
        self.analyze_button.clicked.connect(self.analyze_data)
        self.info_label = QLabel('* select a column or multiple columns before starting analysis')
        self.info_label.setStyleSheet("color: gray")
        buttons_layout.addWidget(self.analyze_button)
        buttons_layout.addWidget(self.info_label)
        buttons_layout.addStretch()
        return buttons_layout

    def create_input_layout(self):
        input_layout = QVBoxLayout()
        spec_layout = self.create_spec_limit_inputs()
        subgroup_layout = self.create_subgroup_size_input()
        bins_layout = self.create_bins_input()
        options_layout = self.create_options_checkboxes()
        buttons_layout = self.create_analyze_button_and_info_label()

        input_layout.addLayout(spec_layout)
        input_layout.addLayout(subgroup_layout)
        input_layout.addLayout(bins_layout)
        input_layout.addLayout(options_layout)
        input_layout.addLayout(buttons_layout)
        return input_layout

    def create_result_table(self):
        self.result_table = CopyableQTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(['Variable', 'Metric', 'Value'])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.result_table.horizontalHeader().setStretchLastSection(False)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.result_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.result_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.result_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.result_table.setSizeAdjustPolicy(QAbstractItemView.AdjustToContents)
        self.result_table.setMinimumHeight(230)
        return self.result_table

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select File', '', 'Data Files (*.xlsx *.xls *.csv)')
        if fname:
            self.file_path.setText(fname)
            self.reset_data()
            try:
                if fname.endswith(('.xlsx', '.xls')):
                    self.is_excel = True
                    self.excel_file = pd.ExcelFile(fname)
                    self.update_sheet_combo()
                    self.load_sheet_data()
                elif fname.endswith('.csv'):
                    self.is_excel = False
                    self.sheet_combo.setEnabled(False)
                    self.data = pd.read_csv(fname)
                    self.display_data()
                else:
                    QMessageBox.warning(self, 'File Error', 'Unsupported file format.')
            except pd.errors.EmptyDataError:
                QMessageBox.critical(self, 'File Error', 'The file is empty.')
            except Exception as e:
                QMessageBox.critical(self, 'File Error', f'Unable to open file:\n{e}')

    def reset_data(self):
        self.data_table.clear()
        self.result_table.setRowCount(0)
        self.is_excel = False
        self.data = None
        self.excel_file = None
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        self.sheet_combo.setEnabled(False)
        self.sheet_combo.blockSignals(False)

    def update_sheet_combo(self):
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        self.sheet_combo.addItems(self.excel_file.sheet_names)
        self.sheet_combo.setEnabled(True)
        self.sheet_combo.setCurrentIndex(0)
        self.sheet_combo.blockSignals(False)

    def load_sheet_data(self):
        if self.is_excel and self.excel_file is not None:
            try:
                sheet_name = self.sheet_combo.currentText()
                if sheet_name:
                    self.data = pd.read_excel(self.excel_file, sheet_name=sheet_name)
                    self.display_data()
            except Exception as e:
                QMessageBox.critical(self, 'Data Error', f'Unable to load data:\n{e}')

    def display_data(self):
        self.data_table.clear()
        self.data_table.setRowCount(self.data.shape[0])
        self.data_table.setColumnCount(self.data.shape[1])
        self.data_table.setHorizontalHeaderLabels(self.data.columns.astype(str))

        self.progress_bar.setMaximum(self.data.shape[0])
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText('Loading data...')

        self.data_table.setUpdatesEnabled(False)
        total_rows = self.data.shape[0]
        progress_update_interval = max(1, total_rows // 100)

        for i in range(total_rows):
            for j in range(self.data.shape[1]):
                item = QTableWidgetItem(str(self.data.iat[i, j]))
                self.data_table.setItem(i, j, item)
            if i % progress_update_interval == 0:
                self.progress_bar.setValue(i + 1)

        self.data_table.setUpdatesEnabled(True)

        self.progress_bar.setVisible(False)
        self.status_label.setText('Data loaded successfully. Please select a column or columns to analyze.')
        self.data_table.resizeColumnsToContents()

    def analyze_data(self):
        lsl_text = self.lsl_input.text()
        usl_text = self.usl_input.text()
        subgroup_size = self.subgroup_input.value()
        remove_zeros = self.remove_zeros_checkbox.isChecked()
        remove_negatives = self.remove_negatives_checkbox.isChecked()
        num_bins = self.bins_input.value()

        if not lsl_text or not usl_text:
            QMessageBox.warning(self, 'Input Error', 'Please enter LSL and USL.')
            return

        try:
            lsl = float(lsl_text)
            usl = float(usl_text)
        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'LSL and USL must be numbers.')
            return

        if not self.check_spec_limits(lsl, usl):
            return

        selected_columns = self.get_selected_columns()
        if not selected_columns:
            QMessageBox.warning(self, 'Selection Error', 'Please select columns to analyze.')
            return

        self.result_table.setRowCount(0)

        for column_name in selected_columns:
            try:
                measurements = self.data[column_name]
                measurements = self.clean_data(measurements, remove_zeros, remove_negatives)
                if measurements.empty:
                    QMessageBox.warning(self, 'Data Error', f'No valid data in column {column_name} after cleaning.')
                    continue
                self.perform_capability_analysis(column_name, measurements, usl, lsl, subgroup_size, num_bins)
            except Exception as e:
                QMessageBox.critical(self, 'Data Error', f'Unable to process data in column {column_name}:\n{e}')

    def get_selected_columns(self):
        selected_columns = []
        selected_ranges = self.data_table.selectedRanges()
        for selected_range in selected_ranges:
            for col in range(selected_range.leftColumn(), selected_range.rightColumn() + 1):
                column_name = self.data_table.horizontalHeaderItem(col).text()
                if column_name not in selected_columns:
                    selected_columns.append(column_name)
        return selected_columns

    def clean_data(self, data, remove_zeros, remove_negatives):
        data = pd.to_numeric(data, errors='coerce')
        data = data.dropna()
        if remove_zeros:
            data = data[data != 0]
        if remove_negatives:
            data = data[data >= 0]
        return data

    def perform_capability_analysis(self, column_name, data, usl, lsl, subgroup_size, num_bins):
        sigma, mean_value = self.estimate_sigma(data, subgroup_size)
        if sigma is None:
            QMessageBox.warning(self, 'Data Error', f'Not enough data in column {column_name} for the specified subgroup size.')
            return
        if sigma == 0 or np.isnan(sigma):
            QMessageBox.warning(self, 'Calculation Error', f'Standard deviation is zero for column {column_name}. Cannot calculate Cp, Cpk.')
            return

        cp, cpk, pp, ppk = self.calculate_process_capability(mean_value, sigma, data, usl, lsl)
        if cp is None:
            QMessageBox.warning(self, 'Calculation Error', f'Standard deviation is zero for column {column_name}. Cannot calculate Cp, Cpk.')
            return

        self.add_results_to_table(column_name, mean_value, sigma, cp, cpk, pp, ppk)
        self.plot_histogram_with_normal_curve(data, mean_value, sigma, lsl, usl, cp, cpk, pp, ppk, num_bins, column_name)

    def estimate_sigma(self, data, subgroup_size):
        if subgroup_size == 1:
            mr = data.diff().abs()
            mr_avg = mr[1:].mean()
            d2 = 1.128  # For n=2
            sigma = mr_avg / d2 if d2 != 0 else 0
            mean_value = data.mean()
        else:
            num_subgroups = len(data) // subgroup_size
            if num_subgroups == 0:
                return None, None
            data = data.iloc[:num_subgroups * subgroup_size]
            subgroups = np.split(data.values, num_subgroups)
            subgroup_means = [np.mean(sg) for sg in subgroups]
            subgroup_ranges = [np.max(sg) - np.min(sg) for sg in subgroups]
            mean_value = np.mean(subgroup_means)
            mean_of_ranges = np.mean(subgroup_ranges)
            d2 = self.calculate_d2(subgroup_size)
            if d2 is None or d2 == 0:
                return None, None
            sigma = mean_of_ranges / d2 if d2 != 0 else 0
        return sigma, mean_value

    def calculate_process_capability(self, mean_value, sigma, data, usl, lsl):
        if sigma == 0 or np.isnan(sigma):
            return None
        cp = (usl - lsl) / (6 * sigma)
        cpu = (usl - mean_value) / (3 * sigma)
        cpl = (mean_value - lsl) / (3 * sigma)
        cpk = min(cpu, cpl)
        pp = (usl - lsl) / (6 * np.std(data, ddof=1))
        ppk = min((usl - mean_value) / (3 * np.std(data, ddof=1)), (mean_value - lsl) / (3 * np.std(data, ddof=1)))
        return cp, cpk, pp, ppk

    def add_results_to_table(self, column_name, mean_value, sigma, cp, cpk, pp, ppk):
        metrics = ['Mean', 'Std Dev (Estimated)', 'Cp', 'Cpk', 'Pp', 'Ppk']
        values = [f"{mean_value:.4f}", f"{sigma:.4f}", f"{cp:.4f}", f"{cpk:.4f}", f"{pp:.4f}", f"{ppk:.4f}"]
        for metric, value in zip(metrics, values):
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)
            self.result_table.setItem(row_position, 0, QTableWidgetItem(column_name))
            self.result_table.setItem(row_position, 1, QTableWidgetItem(metric))
            self.result_table.setItem(row_position, 2, QTableWidgetItem(value))
        self.result_table.resizeColumnsToContents()

    def plot_histogram_with_normal_curve(self, data, mean_value, sigma, lsl, usl, cp, cpk, pp, ppk, num_bins, column_name):
        sns.set(style="whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'figure.figsize': (10, 6),
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'lines.linewidth': 2,
            'lines.markersize': 6,
        })

        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(data, bins=num_bins, density=True, alpha=0.6, color='lightblue', edgecolor='black')

        xmin, xmax = min(data.min(), lsl) - (usl - lsl) * 0.1, max(data.max(), usl) + (usl - lsl) * 0.1
        x = np.linspace(xmin, xmax, 100)
        p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mean_value) ** 2 / (2 * sigma ** 2))
        ax.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')

        ax.axvline(usl, color='red', linestyle='--', linewidth=2, label='USL')
        ax.axvline(lsl, color='blue', linestyle='--', linewidth=2, label='LSL')

        textstr = '\n'.join((
            f'Cp = {cp:.3f}',
            f'Cpk = {cpk:.3f}',
            f'Pp = {pp:.3f}',
            f'Ppk = {ppk:.3f}',
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=props)

        ax.set_title(f'Process Capability Analysis - {column_name}')
        ax.set_xlabel('Measurements')
        ax.set_ylabel('Density')

        ax.legend(loc='upper right')

        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        plt.tight_layout()
        plt.show()

    def calculate_d2(self, n):
        if n <= 1:
            return None
        return np.sqrt(2) * gamma_function(n / 2) / gamma_function((n - 1) / 2)

    def check_spec_limits(self, lsl, usl):
        if lsl >= usl:
            QMessageBox.warning(self, 'Input Error', 'LSL must be less than USL.')
            return False
        return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ProcessCapabilityApp()
    ex.show()
    sys.exit(app.exec_())
