import re
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy_garden.xcamera import XCamera
from kivy.core.window import Window
from kivy.network.urlrequest import UrlRequest
import requests
import os
import time
from kivy.utils import platform

if platform == 'android':
    from android.permissions import request_permissions, Permission, check_permission
    from jnius import autoclass

class CameraApp(App):
    def build(self):
        self.is_capturing = False
        self.capture_type = None

        # Kiểm tra quyền truy cập trước khi tiếp tục
        if platform == 'android' and not self.has_permissions():
            self.request_android_permissions()
            return Label(text="Đang yêu cầu quyền truy cập...")

        layout = BoxLayout(orientation='horizontal')
        self.icon = "icon.jpg"
        self.title = "App Chấm Điểm"

        # Camera layout
        camera_layout = BoxLayout(orientation='vertical', size_hint=(0.8, 1))  # Chiếm 70% chiều rộng của màn hình
        self.camera = XCamera(play=True)
        self.camera.size_hint = (1, 1)
        camera_layout.add_widget(self.camera)

        # Controls layout
        controls_layout = BoxLayout(orientation='vertical', size_hint=(0.2, 1))  # Chiếm 30% chiều rộng của màn hình

        # TextInput để hiển thị kết quả
        self.result_display = TextInput(text='', readonly=True, size_hint=(1, 0.2))
        controls_layout.add_widget(self.result_display)

        self.order_display = TextInput(text='', readonly=True, size_hint=(1, 0.2))
        controls_layout.add_widget(self.order_display)

        # Nút để chụp ảnh đáp án gốc
        self.scan_original_btn = Button(text="Chụp Đáp Án Gốc", size_hint=(1, 0.2))
        self.scan_original_btn.bind(on_press=self.toggle_scan_original)
        controls_layout.add_widget(self.scan_original_btn)

        # Nút để chụp ảnh đáp án sinh viên
        self.scan_student_btn = Button(text="Chụp Đáp Án Sinh Viên", size_hint=(1, 0.2))
        self.scan_student_btn.bind(on_press=self.toggle_scan_student)
        controls_layout.add_widget(self.scan_student_btn)

        # Nút để đọc điểm từ file
        self.read_points_btn = Button(text="Đọc Điểm từ File", size_hint=(1, 0.2))
        self.read_points_btn.bind(on_press=self.read_points_from_file)
        controls_layout.add_widget(self.read_points_btn)

        # Thêm Label để hiển thị trạng thái kết nối
        self.status_label = Label(text='Đang kiểm tra kết nối...', size_hint=(1, 0.2))
        controls_layout.add_widget(self.status_label)

        layout.add_widget(camera_layout)
        layout.add_widget(controls_layout)

        return layout

    def has_permissions(self):
        permissions = [
            Permission.CAMERA,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE,
            Permission.INTERNET,
        ]
        return all(check_permission(p) for p in permissions)

    def request_android_permissions(self):
        def callback(permissions, results):
            if all(results):
                self.restart()
            else:
                print("Permissions not granted.")
        request_permissions(
            [Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE, Permission.INTERNET],
            callback
        )

    def restart(self):
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        context = PythonActivity.mActivity
        packageManager = context.getPackageManager()
        intent = packageManager.getLaunchIntentForPackage(context.getPackageName())
        componentName = intent.getComponent()
        mainIntent = Intent.makeRestartActivityTask(componentName)
        context.startActivity(mainIntent)
        Runtime.getRuntime().exit(0)

    def toggle_scan_original(self, *args):
        self.capture_type = 'original'
        if self.is_capturing:
            self.clear_results()
        else:
            self.scan()

    def toggle_scan_student(self, *args):
        self.capture_type = 'student'
        if self.is_capturing:
            self.clear_results()
        else:
            self.scan()

    def scan(self):
        # Kiểm tra kết nối Internet trước khi thực hiện quét
        self.check_internet_connection(self.on_check_internet)

    def clear_results(self):
        self.result_display.text = ""
        self.order_display.text = ""
        self.scan_original_btn.text = "Chụp Đáp Án Gốc"
        self.scan_student_btn.text = "Chụp Đáp Án Sinh Viên"
        self.is_capturing = False

    def capture_image(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"goc.jpg" if self.capture_type == 'original' else f"sinhvien.jpg"
        filepath = os.path.join(self.get_gallery_path(), filename)
        self.camera.export_to_png(filepath)
        print(f"Image saved to {filepath}")

        # Gửi ảnh lên máy chủ để chấm điểm
        self.send_image_to_server(filepath)

    def get_gallery_path(self):
        if platform == 'android':
            from android.storage import primary_external_storage_path
            gallery_path = os.path.join(primary_external_storage_path(), 'DCIM', 'CameraApp')
        else:
            gallery_path = os.path.join(os.path.expanduser('~'), 'Pictures')
        if not os.path.exists(gallery_path):
            os.makedirs(gallery_path)
        return gallery_path

    def send_image_to_server(self, filepath):
        if self.capture_type == 'original':
            api_url = 'http://192.168.100.165:8000/origin/'
        else:
            api_url = 'http://192.168.100.165:8000/auto_grade/'  # Thay đổi endpoint cho đáp án sinh viên

        files = {'image': open(filepath, 'rb')}
        try:
            response = requests.post(api_url, files=files)
            if response.status_code == 200:
                data = response.json()
                self.result_display.text = f"Extracted Text: {data.get('text', 'N/A')}"
            else:
                self.status_label.text = f"Failed to upload image: {response.status_code}"
                print(f"Failed to upload image: {response.status_code}")
        except Exception as e:
            self.status_label.text = f"Failed to send image to server: {e}"
            print(f"Failed to send image to server: {e}")

    def check_internet_connection(self, callback):
        UrlRequest('http://clients3.google.com/generate_204', on_success=callback, on_error=self.no_connection, on_failure=self.no_connection)

    def on_check_internet(self, request, result):
        self.status_label.text = 'Connected'
        self.capture_image()

    def no_connection(self, *args):
        self.status_label.text = 'No Internet Connection'
        print('No Internet Connection')

    def read_points_from_file(self, *args):
        filepath = os.path.join(self.get_gallery_path(), 'points.txt')
        if not os.path.exists(filepath):
            self.result_display.text = "File không tồn tại."
            return

        try:
            with open(filepath, 'r') as file:
                content = file.read()
                points = self.extract_points(content)
                if points:
                    self.result_display.text = f"Số điểm: {points}"
                else:
                    self.result_display.text = "Không tìm thấy điểm."
        except Exception as e:
            self.result_display.text = f"Lỗi khi đọc file: {e}"
            print(f"Lỗi khi đọc file: {e}")

    def extract_points(self, text):
        match = re.search(r'ABCDPOINT:\s*(\d+)', text)
        if match:
            return match.group(1)
        return None

if __name__ == '__main__':
    CameraApp().run()
