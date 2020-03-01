# -*- coding: utf-8 -*-
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../ui_with_tkinter'))
sys.path.append(os.path.join(BASE_DIR, '../face_recognition'))
import classifier as clf
import numpy as np
import cv2
import tensorflow as tf
import facenet
import detect_face
import align_dlib
import dlib
from PIL import Image, ImageDraw, ImageFont

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True  # allocate dynamically


def predict_labels(best_model_list,
                   test_features,
                   all_data=False):
    """Predict with the best three models.
    Inputs:
      - best_model_list: A list contains the best three models.
      - test_features: A numpy array contains the features of the test images.
      - face_number: An integer, the number of faces that we found in the test image.
      - all_data: True or False, whether the model is trained with all data.

    Return:
      - labels_array: A numpy array contains the labels which predicted by
            the best three models .
    """
    labels_array = []
    for index, model in enumerate(best_model_list):
        if all_data:
            predict = model[0].predict(test_features)
        else:
            predict = model[0].predict(test_features)
        labels_array.append(predict)

    labels_array = np.array(labels_array)
    return labels_array


def predict_probability(best_probability_model,
                        test_features,
                        face_number,
                        all_data=False):
    """Predict with softmax probability model.
    Inputs:
      - best_probability_model: The best pre_trained softmax model.
      - test_features: A numpy array contains the features of the test images.
      - face_number: An integer, the number of faces that we found in the test image.
      - all_data: True or False, whether the model is trained with all data.

    Returns:
      - labels: A numpy array contains the predicted labels.
      - probability: A numpy array contains the probability of the label.
    """
    labels, probability = None, None
    if face_number is not 0:
        if all_data:
            labels = best_probability_model[0].predict(test_features)
            all_probability = best_probability_model[0].predict_proba(test_features)
        else:
            labels = best_probability_model[0].predict(test_features)
            all_probability = best_probability_model[0].predict_proba(test_features)
        print('all_probability', all_probability)
        print('labels', labels)
        print('all_probability.shape', all_probability.shape)

        # print(probability.shape) -> (face_number, class_number)
        # print(all_probability)
        probability = all_probability[np.arange(face_number), np.argmax(all_probability, axis=1)]
        # np.argmax(all_probability, axis=1)]
        print('probability', probability)
        print('probability.shape', probability.shape)

    return labels, probability


def find_people_from_image(best_probability_model, test_features, face_number, all_data=False):
    """
    Inputs:
      - best_probability_model: The best pre_trained softmax model.
      - test_features: A numpy array contains the features of the test images.
      - face_number: An integer, the number of faces that we found in the test image.
      - all_data: If all_data is True, using the models which trained by all
          training data. Otherwise using the models which trained by partial data.

    e.g.
    test_features: [feature_1, feature_2, feature_3, feature_4, feature_5]

    first we get all predictive labels and their corresponding probability
    [(A, 0.3), (A, 0.2), (B, 0.1), (B, 0.3), (C, 0.1)]
    second we choose the maximum probability for each unique label
    [(A, 0.3), (B, 0.3), (C, 0.1)]
    finally, get the indices for each unique label.
    [0, 3, 4]

    then return
    label:
    probability:
    unique_labels: [A, B, C]
    unique_probability: [0.3, 0.3, 0.1]
    unique_indices: [0, 3, 4]
    """
    labels, probability = predict_probability(
        best_probability_model, test_features, face_number, all_data=all_data)

    unique_labels = np.unique(labels)
    print('unique_labels', unique_labels)
    unique_probability, unique_indices = [], []

    for label in unique_labels:
        indices = np.argwhere(labels == label)[:, 0]
        unique_probability.append(np.max(probability[indices]))
        unique_indices.append(indices[np.argmax(probability[indices])])

    unique_probability = np.array(unique_probability)
    unique_indices = np.array(unique_indices)
    print('unique_probability', unique_probability)
    print('unique_indices', unique_indices)

    return labels, probability, unique_labels, unique_probability, unique_indices


def check_sf_features(feature, label):
    """Verification.
    Inputs:
       - feature: The feature to be verified.
       - label: The label used for verification.

    Returns:
       - True or False, verification result.
       - sum_dis, the distance loss.
    """
    if os.path.exists('features'):
        sf_features = np.load(open('features/{}.npy'.format(label), 'rb'))
    else:
        sf_features = np.load(open('../features/{}.npy'.format(label), 'rb'))
    sf_dis = np.sqrt(np.sum((sf_features - feature) ** 2, axis=1))
    # print(sf_dis)
    sum_dis = np.sum(sf_dis)
    # print(sum_dis)
    valid_num1 = np.sum(sf_dis > 1.0)
    valid_num2 = np.sum(sf_dis > 1.1)
    if valid_num1 >= 4:
        return False, sum_dis

    if valid_num2 >= 3:
        return False, sum_dis

    return True, sum_dis


def recognition(image, state, fr=None, all_data=False, language='english'):
    """Implement face verification, face recognition and face search functions.
    Inputs:
      - image_path: A string contains the path to the image.
      - fr: The object of the UI class.
      - state: 'verification', face verification
               'recognition', face recognition
               'search', face search.
      - used_labels: The labels used to face verification and face search,
            which does not used in face recognition
      - image_size: The input size of the MTCNN.
      - all_data: If all_data is True, using the models which trained by all
            training data. Otherwise using the models which trained by partial data.
      - output: True or False, output the process information.

    Returns:
      - answer: The answer predicted by the model.
      - image_data: The image data after prediction.
    """
    answer = ''
    predict_info = []
    for i in state:
        if i not in {'verification', 'recognition', 'search'}:
            raise ValueError('{} is not a valid argument!'.format(state))
    test_features, face_number, face_boxes, flag = process('test',
                                                           image,
                                                           image_size=144, )
    if flag:
        for i in state:
            if i == 'verification':
                print('Start verification')

                labels_array = predict_labels(best_classifier_model_list,
                                              test_features,
                                              all_data=all_data)
                labels = []
                for line in labels_array.T:  # 转置
                    unique, counts = np.unique(line, return_counts=True)  # 该函数是去除数组中的重复数字，并进行排序之后输出
                    temp_label = unique[np.argmax(counts)]
                    labels.append(temp_label)

                if used_labels[0] in labels:
                    if language == 'chinese':
                        answer = answer + '验证成功！这张图像被认定为{}！'.format(used_labels[0])
                        info = '验证成功！这张图像被认定为{}！'.format(used_labels[0])
                    else:
                        answer = answer + 'Successful Verification! This image was ' \
                                          'identified as {}!'.format(used_labels[0])
                        info = 'Successful Verification! This image ' \
                               'was identified as {}!'.format(used_labels[0])
                else:
                    if language == 'chinese':
                        answer = answer + '验证失败！这张图像不被认定为{}！' \
                                          ''.format(used_labels[0])
                        info = '验证失败！这张图像不被认定为{}！'.format(used_labels[0])
                    else:
                        answer = answer + 'Verification failed! This image is not ' \
                                          'recognized as {}!'.format(used_labels[0])
                        info = 'Verification failed! This image is not ' \
                               'recognized as {}!'.format(used_labels[0])

                for index, box in enumerate(face_boxes):
                    face_position = box.astype(int)
                    cv2.rectangle(image, (face_position[0], face_position[1]), (
                        face_position[2], face_position[3]), (0, 255, 0), 2)

                if fr is not None:
                    predict_info.append(info)
                    fr.show_information(predict_info, predict=True)
                if mode == 'video':
                    cv2.imshow('camera', image)
                else:
                    cv2.imshow('camera', image)
                    cv2.imwrite('../result/verification.jpg', image)
                    cv2.waitKey()

            elif i == 'recognition':
                print('Start recognition')
                labels, _, unique_labels, unique_probability, unique_indices = \
                    find_people_from_image(best_probability_model, test_features, face_number, all_data=all_data)
                info = ''
                if language == 'chinese':
                    info = info + '从图像中检测到'
                else:
                    info = info + 'Detected '
                answer = answer + info

                for index, label in enumerate(labels):
                    if index in unique_indices:
                        if check_sf_features(test_features[index], label)[0] is False:
                            if language == 'chinese':
                                # labels[index] = '未知'
                                labels[index] = ''
                            else:
                                # labels[index] = 'Unknown'
                                labels[index] = ''
                        else:
                            if language == 'chinese':
                                info = info + '{}，'.format(label)
                                answer = answer + '{}，'.format(label)
                            else:
                                info = info + '{},'.format(label)
                                answer = answer + '{},'.format(label)
                    else:
                        if language == 'chinese':
                            labels[index] = ''
                            # labels[index] = '未知'
                        else:
                            # labels[index] = 'Unknown'
                            labels[index] = ''
                info = info[:-1]
                answer = answer[:-1]
                if language == 'english':
                    info = info + ' in this image!'
                    answer = answer + ' in this image!'
                else:
                    info = info + '！'
                    answer = answer + '！'

                if fr is not None:
                    predict_info.append(info)
                    fr.show_information(predict_info, predict=True)
                for index, label in enumerate(labels):
                    face_position = face_boxes[index].astype(int)
                    image_data = puttext(image, face_position, label, face_number, language='english')
                if mode == 'video':
                    cv2.imshow('camera', image_data)
                else:
                    cv2.imshow('camera', image_data)
                    cv2.imwrite('../result/recognition.jpg', image_data)
                    cv2.waitKey()


            elif i == 'search':
                print('Start search')
                _, _, unique_labels, unique_probability, unique_indices = \
                    find_people_from_image(best_probability_model, test_features, face_number, all_data=all_data)
                n = unique_labels.shape[0]

                found_indices = []
                if language == 'chinese':
                    info = '从图像中找到'
                else:
                    info = 'Found '
                answer = answer + info
                for i in range(n):
                    if unique_labels[i] not in used_labels:
                        continue
                    index = unique_indices[i]
                    if check_sf_features(test_features[index], unique_labels[i])[0] is False:
                        continue
                    if language == 'chinese':
                        answer = answer + '{}，'.format(unique_labels[i])
                        info = info + '{}，'.format(unique_labels[i])
                    else:
                        answer = answer + '{},'.format(unique_labels[i])
                        info = info + '{},'.format(unique_labels[i])
                    found_indices.append(i)

                info = info[:-1]
                answer = answer[:-1]
                if language == 'english':
                    info = info + ' in this image!'
                    answer = answer + ' in this image!'
                else:
                    info = info + '！'
                    answer = answer + '！'

                if fr is not None:
                    predict_info.append(info)
                    fr.show_information(predict_info, predict=True)
                for i in found_indices:
                    index = unique_indices[i]
                    face_position = face_boxes[index].astype(int)
                    image_data = puttext(image, face_position, unique_labels[i], face_number, language='english')
                    if mode == 'video':
                        cv2.imshow('camera', image_data)
                    else:
                        cv2.imshow('camera', image_data)
                        cv2.imwrite('../result/search.jpg', image_data)
                        cv2.waitKey()
            return answer
    else:
        return 0


def process(state, image, image_size=144):
    if state == 'test':
        test_image_data = image.copy()
        test_features = []
        face_boxes, _ = detect_face.detect_face(
            test_image_data, minsize, p_net, r_net, o_net, threshold, factor)
        face_number = face_boxes.shape[0]

        if face_number is 0:
            print('face number is 0')
            return None, face_number, None, 0
        else:
            for face_position in face_boxes:
                face_position = face_position.astype(int)
                face_rect = dlib.rectangle(int(face_position[0]), int(
                    face_position[1]), int(face_position[2]), int(face_position[3]))

                # test_pose_landmarks = face_pose_predictor(test_image_data, face_rect)
                # test_image_landmarks = test_pose_landmarks.parts()

                aligned_data = face_aligner.align(
                    image_size, test_image_data, face_rect, landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)

                # plt.subplot(face_number, 1, index)
                # plt.imshow(aligned_data)
                # plt.axis('off')
                # plt.show()
                # cv2.imwrite('datasets/team_aligned/{}.jpg'.format(str(index)),
                #             cv2.cvtColor(aligned_data, cv2.COLOR_RGB2BGR))

                aligned_data = facenet.prewhiten(aligned_data)
                last_data = aligned_data.reshape((1, image_size, image_size, 3))
                test_features.append(sess.run(embeddings, feed_dict={
                    images_placeholder: last_data, phase_train_placeholder: False})[0])
            test_features = np.array(test_features)

            return test_features, face_number, face_boxes, 1


def puttext(image_data, face_position, label, face_number, language='english'):
    # print('face_position[%d]' % (index), face_position)
    label_pixels, font_size, line_size, rect_size, up_offset = \
        None, None, None, None, None
    if face_number == 1:
        rect_size = 2
        if language == 'chinese':
            label_pixels = 30 * len(label)  # 140 * len(label)
            font_size = 30  # 140
            up_offset = 40  # 140
        else:
            label_pixels = 20 * len(label)
            up_offset = 20
            font_size = 1
        line_size = 2
    elif face_number < 4:
        rect_size = 2  # 7
        if language == 'chinese':
            label_pixels = 30 * len(label)  # 140 * len(label)
            font_size = 30  # 140
            up_offset = 40  # 140
        else:
            label_pixels = 20 * len(label)
            up_offset = 20
            font_size = 1
        line_size = 2
    elif face_number >= 4:
        rect_size = 2  # 6
        if language == 'chinese':
            label_pixels = 30 * len(label)  # 100 * len(label)
            font_size = 30  # 100
            up_offset = 40  # 100
        else:
            label_pixels = 10 * len(label)
            up_offset = 20
            font_size = 1
        line_size = 2

    dis = (label_pixels - (face_position[2] - face_position[0])) // 2
    # dis = 0

    if language == 'chinese':
        # The color coded storage order in cv2(BGR) and PIL(RGB) is different
        cv2img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2img)
        # use PIL to display Chinese characters in images
        draw = ImageDraw.Draw(pilimg)
        font = ImageFont.truetype('../resources/STKAITI.TTF',
                                  font_size, encoding="utf-8")
        # draw.text((face_position[0] - dis, face_position[1]-up_offset),
        # label, (0, 0, 255), font=font)
        draw.text((face_position[0] - dis, face_position[1] - up_offset),
                  label, (0, 0, 255), font=font)
        # convert to cv2
        image_data = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(image_data, label,
                    (face_position[0] - dis, face_position[1] - up_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (0, 0, 255), line_size)
    image_data = cv2.rectangle(image_data, (face_position[0], face_position[1]),
                               (face_position[2], face_position[3]),
                               (0, 255, 0), rect_size)
    return image_data


if __name__ == '__main__':
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    gpu_memory_fraction = 0.6

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False))
        with sess.as_default():
            p_net, r_net, o_net = detect_face.create_mtcnn(sess, '../models/mtcnn/')
    predictor_model = '../models/shape_predictor_68_face_landmarks.dat'
    face_aligner = align_dlib.AlignDlib(predictor_model)
    model_dir = '../models/20170512-110547/20170512-110547.pb'  # model directory
    tf.Graph().as_default()
    sess = tf.Session()
    facenet.load_model(model_dir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    all_data = True
    best_classifier_model_list = clf.load_best_classifier_model(all_data=all_data)
    best_probability_model = clf.load_best_probability_model(all_data=True)
    #print('best_probability_model', best_probability_model)

    fit_all_data = True
    mode = 'video'  # 'video' or 'picture'
    state = ['recognition']  # state = ['verification', 'recognition', 'search']
    used_labels = ['LeBron']  # verification只验证used_labels[0],search时会查找所有label

    if mode == 'video':
        video = "http://admin:admin@192.168.0.13:8081"
        # video = 0
        capture = cv2.VideoCapture(video)
        cv2.namedWindow("camera", 1)
        language = 'english'
        c = 0
        num = 0
        frame_interval = 3  # frame intervals
        test_features = []
        while True:
            ret, frame = capture.read()
            cv2.imshow("camera", frame)
            answer = recognition(frame, state, fr=None, all_data=fit_all_data, language='english', )
            print(answer)
            c += 1
            key = cv2.waitKey(3)
            if key == 27:
                # esc键退出
                print("esc break...")
                break
            if key == ord(' '):
                # 保存一张图像
                num = num + 1
                filename = "frames_%s.jpg" % num
                cv2.imwrite('../result/' + filename, frame)
        # When everything is done, release the capture
        capture.release()
        cv2.destroyWindow("camera")
    else:
        image = cv2.imread('../test/42.jpg')
        answer = recognition(image, fr=None, state=state, all_data=fit_all_data, language='english', )
        print(answer)
