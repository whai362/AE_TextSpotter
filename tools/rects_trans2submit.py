import mmcv
import numpy as np

preds = mmcv.load('results_test.json')
preds = sorted(preds, key=lambda x: x['img_name'])
submit = ''

def is_clockwise(points):
    points = np.array(points)
    cp_postive = 0
    for i in range(len(points)):
        cp = (points[i, 0] * points[(i + 1) % len(points), 1]) - (points[i, 1] * points[(i + 1) % len(points), 0])
        cp_postive += cp
    return cp_postive > 0

for pred in preds:
    img_name = pred['img_name'].replace('ReCTS_task3_and_task_4_', '')
    bboxes = pred['points']
    scores = pred['scores']
    texts = pred['texts']
    submit += img_name + '\n'
    for bbox, score, text in zip(bboxes, scores, texts):
        if not is_clockwise(bbox):
            bbox = bbox[::-1]
        if score > 0.60 and len(text) > 0:
            submit += ','.join([str(max(0, int(x[0]))) + ',' + str(max(0, int(x[1]))) for x in bbox]) + ',' + text + '\n'

with open('submit.txt','w') as f:
    f.write(submit)