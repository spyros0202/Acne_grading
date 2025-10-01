from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import numpy as np
import json
import joblib
import os
# from django.contrib.auth.decorators import login_required

from classifier import module_feat_generation2 as fg2
from classifier import module_feat_generation3 as fg3

# @login_required

def classify_image(request):
    if request.method == 'POST':
        model_path1 = os.path.join('static/', 'decision_tree-model0-1_2-3(new).pkl')
        scaler_path1 = os.path.join('static/', 'scaler0-1_2-3(new).pkl')
        model_path2 = os.path.join('static/', 'decision_tree-model1_2-3(new).pkl')
        scaler_path2 = os.path.join('static/', 'scaler1_2-3(new).pkl')

        model1 = joblib.load(model_path1)
        scaler1 = joblib.load(scaler_path1)
        model2 = joblib.load(model_path2)
        scaler2 = joblib.load(scaler_path2)

        use_matrix = request.POST.get("matrix", "false").lower() == "true"
        img_file = request.FILES['image']
        image = Image.open(img_file).convert('L')
        img_np = np.array(image)
        results = {}

        try:
            rois = json.loads(request.POST.get('rois'))
            if use_matrix:
                if not rois:
                    return JsonResponse({'error': 'No ROI to center the matrix on.'})

                # Extract center from the first ROI
                x1, y1, x2, y2 = map(int, rois[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                roi_size = 50
                rows, cols = 5, 5
                start_x = center_x - (cols // 2) * roi_size
                start_y = center_y - (rows // 2) * roi_size

                matrix_rois = []
                healthy_count = low_count = high_count = 0
                roi_index = 0

                for row in range(rows):
                    for col in range(cols):
                        x = start_x + col * roi_size
                        y = start_y + row * roi_size

                        if x < 0 or y < 0 or x + roi_size > img_np.shape[1] or y + roi_size > img_np.shape[0]:
                            continue

                        roi = img_np[y:y + roi_size, x:x + roi_size]
                        f1 = np.array(fg3.generate_2ndOrder_RunLength_matrix_features(roi)[1])
                        f2 = np.array(fg3.generate_local_binary_pattern_featuresLBP1(roi)[1])
                        f3 = np.array(fg3.generate_2ndOrder_coocurrence_matrix_features(roi)[1])
                        f4 = np.array(fg3.generate_first_order_features_py(roi)[1])
                        class1 = np.concatenate((f1, f2, f3, f4)).reshape(1, -1)

                        pred1 = model1.predict(scaler1.transform(class1))[0]

                        if pred1 == 0:
                            label = "Healthy"
                            healthy_count += 1
                        else:
                            f11 = np.array(fg2.generate_local_binary_pattern_featuresLBP1(roi)[1])
                            f22 = np.array(fg2.generate_local_binary_pattern_featuresLBP4(roi)[1])
                            f33 = np.array(fg2.generate_2ndOrder_coocurrence_matrix_features1(roi)[1])
                            f44 = np.array(fg2.generate_2ndOrder_coocurrence_matrix_features2(roi)[1])
                            class2 = np.concatenate((f11, f22, f33, f44)).reshape(1, -1)
                            pred2 = model2.predict(scaler2.transform(class2))[0]
                            if pred2 == 0:
                                label = "Low Level"
                                low_count += 1
                            else:
                                label = "High Level"
                                high_count += 1

                        results[roi_index] = label
                        matrix_rois.append([x, y, x + roi_size, y + roi_size])
                        roi_index += 1

                total = healthy_count + low_count + high_count
                return JsonResponse({
                    "results": results,
                    "rois": matrix_rois,
                    "summary": {
                        "Healthy": round(healthy_count / total * 100, 1) if total else 0,
                        "Low Level": round(low_count / total * 100, 1) if total else 0,
                        "High Level": round(high_count / total * 100, 1) if total else 0,
                    }
                })
            else:
                for i, (x1, y1, x2, y2) in enumerate(rois):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    roi = img_np[y1:y2, x1:x2]
                    f1 = np.array(fg3.generate_2ndOrder_RunLength_matrix_features(roi)[1])
                    f2 = np.array(fg3.generate_local_binary_pattern_featuresLBP1(roi)[1])
                    f3 = np.array(fg3.generate_2ndOrder_coocurrence_matrix_features(roi)[1])
                    f4 = np.array(fg3.generate_first_order_features_py(roi)[1])
                    class1 = np.concatenate((f1, f2, f3, f4)).reshape(1, -1)

                    pred1 = model1.predict(scaler1.transform(class1))[0]
                    if pred1 == 0:
                        results[i] = "Healthy"
                    else:
                        f11 = np.array(fg2.generate_local_binary_pattern_featuresLBP1(roi)[1])
                        f22 = np.array(fg2.generate_local_binary_pattern_featuresLBP4(roi)[1])
                        f33 = np.array(fg2.generate_2ndOrder_coocurrence_matrix_features1(roi)[1])
                        f44 = np.array(fg2.generate_2ndOrder_coocurrence_matrix_features2(roi)[1])
                        class2 = np.concatenate((f11, f22, f33, f44)).reshape(1, -1)
                        pred2 = model2.predict(scaler2.transform(class2))[0]
                        results[i] = "Low Level" if pred2 == 0 else "High Level"

                return JsonResponse({"results": results, "rois": rois})

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return render(request, 'upload.html')
