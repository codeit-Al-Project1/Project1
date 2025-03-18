import zipfile
import os

if __name__ == "__main__":

    zip_file_path = "../data/ai01-level1-project.zip"
    extract_path = "../data"

    # 압축 파일 존재 여부 확인
    if not os.path.exists(zip_file_path):
        print(f"오류: 압축 파일 '{zip_file_path}'을 찾을 수 없습니다.")
    else:
        try:
            # 압축 해제
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"'{zip_file_path}' 파일의 압축을 해제하여 '{extract_path}'에 저장했습니다.")
        except Exception as e:
            print(f"오류: 압축 해제 중 오류가 발생했습니다. {e}")