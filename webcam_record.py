import cv2

# 웹캠 비디오 녹화
def record_webcam_video(width=224, height=224):
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    
    # 웹캠이 성공적으로 열렸는지 확인
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return
    
    # 비디오 코덱 및 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    recording = False
    video_count = 0
    
    print("사용법:")
    print("'c' : 비디오 캡처 시작/중지")
    print("'q' : 프로그램 종료")
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        # 프레임 읽기 실패 시 종료
        if not ret:
            print("오류: 프레임을 읽을 수 없습니다.")
            break
        
        # 프레임 크기 조정
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # 녹화 상태 표시
        display_frame = resized_frame.copy()
        if recording:
            cv2.putText(display_frame, "REC", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 화면에 프레임 표시
        cv2.imshow('Webcam Video Capture', display_frame)
        
        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        
        # 'c' 키: 녹화 시작/중지
        if key == ord('c'):
            if not recording:
                # 녹화 시작
                video_count += 1
                output_path = f'recorded_video_{video_count}.avi'
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                recording = True
                print(f"녹화 시작: {output_path}")
            else:
                # 녹화 중지
                out.release()
                recording = False
                print("녹화 중지")
        
        # 녹화 중인 경우 프레임 저장
        if recording and out is not None:
            out.write(resized_frame)
        
        # 'q' 키: 프로그램 종료
        if key == ord('q'):
            break
    
    # 리소스 해제
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")

def main():
    record_webcam_video()

if __name__ == '__main__':
    main()