from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import numpy as np
import json
import joblib
import os
import pandas as pd

from classifier import module_feat_generation2 as fg2
from classifier import module_feat_generation3 as fg3


def save_features_as_dataframe(features, feature_names):
    return pd.DataFrame(features, columns=feature_names)

def classify_image(request):
    if request.method == 'POST':
        img_file = request.FILES['image']
        image = Image.open(img_file).convert('L')
        img_np = np.array(image)
        rois = json.loads(request.POST.get('rois'))

        results = {}
        try:
            # Load models and scalers
            model_path1 = os.path.join('static/', 'decision_tree-model0-1_2-3(new).pkl')
            scaler_path1 = os.path.join('static/', 'scaler0-1_2-3(new).pkl')
            model_path2 = os.path.join('static/', 'decision_tree-model1_2-3(new).pkl')
            scaler_path2 = os.path.join('static/', 'scaler1_2-3(new).pkl')

            model1 = joblib.load(model_path1)
            scaler1 = joblib.load(scaler_path1)
            model2 = joblib.load(model_path2)
            scaler2 = joblib.load(scaler_path2)

            for i, (x1, y1, x2, y2) in enumerate(rois):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                roi = img_np[y1:y2, x1:x2]
                # roi_img = Image.fromarray(roi)
                #
                # # Define save path
                # save_dir = os.path.join(os.path.dirname(__file__), 'roi_images')
                # os.makedirs(save_dir, exist_ok=True)
                #
                # # Save with a unique filename
                # roi_img.save(os.path.join(save_dir, f'roi_{i}.png'))
                # First-level features
                feats_TMR, f1 = fg3.generate_2ndOrder_RunLength_matrix_features(roi)
                feats_LBP4, f2 = fg3.generate_local_binary_pattern_featuresLBP1(roi)
                feats_DIS, f3 = fg3.generate_2ndOrder_coocurrence_matrix_features(roi)
                feats_MEAN, f4 = fg3.generate_first_order_features_py(roi)

                f1 = np.array(f1)
                f2 = np.array(f2)
                f3 = np.array(f3)
                f4 = np.array(f4)
                class1 = np.concatenate((f1, f2, f3, f4), axis=0).reshape(1, -1)

                data1 = scaler1.transform(class1)
                pred1 = model1.predict(data1)[0]

                if pred1 == 0:
                    results[i] = "Healthy"
                else:
                    # Second-level features
                    feats_LBP5, f11 = fg2.generate_local_binary_pattern_featuresLBP1(roi)
                    feats_CON_MEAN, f22 = fg2.generate_local_binary_pattern_featuresLBP4(roi)
                    feats_LBP1, f33 = fg2.generate_2ndOrder_coocurrence_matrix_features1(roi)
                    feats_CON_RANGE, f44 = fg2.generate_2ndOrder_coocurrence_matrix_features2(roi)

                    f11 = np.array(f11)
                    f22 = np.array(f22)
                    f33 = np.array(f33)
                    f44 = np.array(f44)
                    class2 = np.concatenate((f11, f22, f33, f44), axis=0).reshape(1, -1)

                    data2 = scaler2.transform(class2)
                    pred2 = model2.predict(data2)[0]

                    if pred2 == 0:
                        results[i] = "Low Level"
                    else:
                        results[i] = "High Level"

        except Exception as e:
            return JsonResponse({'error': str(e)})

        return JsonResponse(results)

    return render(request, 'upload.html')
