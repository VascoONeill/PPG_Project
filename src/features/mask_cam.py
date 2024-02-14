import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)


def CapturePPG(Sample=0):
    # Initiate the capture
    cap = cv2.VideoCapture(Sample)

    # Extract the FPS
    # fps = cv2.CAP_PROP_FPS

    timestamp = []

    # Initiate face_mesh
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        # Definition of the mask that will contain the mean RGB value inside the ROI
        mask_mean_RGB = []

        while True:

            success, image = cap.read()

            # Making sure the video is getting captured
            if not success:
                break

            # Detecting the image
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:




                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=my_drawing_specs
                    )

                    roi = [425, 266, 330, 347, 346, 352, 411, 425, 280]
                    # Region of Interest (ROI) insert the face landmarks you want to detect
                    roi_points = [(int(face_landmarks.landmark[idx].x * image.shape[1]),
                                   int(face_landmarks.landmark[idx].y * image.shape[0])) for idx in roi]

                    # Create a mask for the nose region
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(roi_points)], 255)  # White color

                    # Ensure the mask is not empty
                    if not mask.any():
                        continue

                    # Compute the average color within the nose region
                    average_color = cv2.mean(image, mask)

                    # Display the average color information (not necessary)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, f'Average Color: {average_color[:3]}', (10, 30), font, 0.7, (255, 255, 255), 2)

                    # Apply the mask to the original image
                    roi = cv2.bitwise_and(image, image, mask=mask)

                    # Display the original image with ROI
                    cv2.imshow("My video capture", cv2.addWeighted(image, 0.5, roi, 1, 0))

                    mask_mean_RGB.append(average_color[:3])

                    timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))

            else:
                cv2.putText(image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.namedWindow("My video capture", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("My video capture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("My video capture", cv2.flip(image, 1))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return mask_mean_RGB, timestamp


def video_processing(video_sample=0):

    signal, timestamp = CapturePPG(video_sample)

    data = np.array(signal)

    r_signal = data[:, 0]
    g_signal = data[:, 1]
    b_signal = data[:, 2]

    timestamp = np.array(timestamp)
    time_sec = timestamp/(1*(10**3))

    return r_signal, g_signal, b_signal, time_sec
