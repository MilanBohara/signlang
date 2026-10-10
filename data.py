from function import *
from time import sleep

image_root = os.path.join(os.path.dirname(__file__), 'Image')
# Only process letters that have all required images; skip the rest
ready_actions = []
for action in actions:
    missing = [
        sequence for sequence in range(no_sequences)
        if not os.path.isfile(os.path.join(image_root, action, f'{sequence}.png'))
    ]
    if missing:
        print(f'Skipping {action}: missing {len(missing)} of {no_sequences} images. '
              'Run collectdata.py to capture more samples.')
    else:
        ready_actions.append(action)
if not ready_actions:
    raise FileNotFoundError(
        f'No letter has all {no_sequences} images in {image_root}. '
        'Run collectdata.py to capture samples first.'
    )
print('Processing:', ' '.join(ready_actions))

for action in ready_actions:
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Single-image detectors used when the tracker misses a hand. Fists and
# side-on hands are hard for MediaPipe, so try both model sizes at 1x and 2x.
static_detectors = [
    mp_hands.Hands(static_image_mode=True, model_complexity=complexity, min_detection_confidence=0.3)
    for complexity in (1, 0)
]

def detect_with_fallback(frame):
    for scale in (1, 2):
        scaled = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame
        for detector in static_detectors:
            _, results = mediapipe_detection(scaled, detector)
            if results.multi_hand_landmarks:
                return results
    return None

# cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    no_hand = {action: [] for action in ready_actions}
    # NEW LOOP
    # Loop through actions
    for action in ready_actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            fallback_results = None
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                # ret, frame = cap.read()
                frame = cv2.imread(os.path.join(image_root, action, f'{sequence}.png'))
                # frame=cv2.imread('{}{}.png'.format(action,sequence))
                # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                # Make detections
                image, results = mediapipe_detection(frame, hands)
#                 print(results)
                if not results.multi_hand_landmarks:
                    if fallback_results is None:
                        fallback_results = detect_with_fallback(frame) or results
                    results = fallback_results

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                if frame_num == sequence_length - 1 and not keypoints.any():
                    no_hand[action].append(sequence)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    # cap.release()
    cv2.destroyAllWindows()

for detector in static_detectors:
    detector.close()

# Report images where no hand was found; these are skipped by trainmodel.py
for action in ready_actions:
    found = no_sequences - len(no_hand[action])
    print(f'{action}: hand found in {found}/{no_sequences} images', end='')
    if no_hand[action]:
        print(f' - no hand in: {", ".join(f"{s}.png" for s in no_hand[action])}', end='')
        if found < 20:
            print(' (recapture this letter with the whole hand inside the box)', end='')
    print()


