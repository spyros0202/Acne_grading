from django.shortcuts import render
from django.http import JsonResponse
# from .utils.image_analysis import classify_rois  # To be created
from PIL import Image
import numpy as np
import json


def classify_image(request):
    return render(request, 'upload.html')

# def upload_image(request):
#     if request.method == 'POST':
#         img_file = request.FILES['image']
#         image = Image.open(img_file).convert('L')
#         img_np = np.array(image)
#         rois = json.loads(request.POST.get('rois'))
#
#         result = classify_rois(img_np, rois)
#         return JsonResponse(result)
#
#     return render(request, 'classifier/upload.html')
