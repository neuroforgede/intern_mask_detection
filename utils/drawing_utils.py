import tensorflow as tf
from PIL import ImageDraw
import os


def draw_parameter():
    parameter = {"use_rnd_color": False,
                 "default_color_0": (0, 255, 0),
                 "default_color_1": (255, 0, 0)}
    return parameter


def draw_all_pictures_of_one_batch(step, batch_size, output_dir, array_of_all_images,
                                   cropped_person_for_all_images_of_global_batch):
    for i in range(len(array_of_all_images)):

        parameter = draw_parameter()

        if parameter["use_rnd_color"]:
            colors = tf.random.uniform((2, 4), maxval=256, dtype=tf.int32)
        else:
            colors = (parameter["default_color_0"], parameter["default_color_1"])

        draw = ImageDraw.Draw(array_of_all_images[i])
        for j in range(len(cropped_person_for_all_images_of_global_batch[i])):
            rectangle = cropped_person_for_all_images_of_global_batch[i][j][0]
            y1, x1, y2, x2 = tf.split(rectangle, 4)

            label_id = round(cropped_person_for_all_images_of_global_batch[i][j][1])
            if label_id == 0:
                label = 'person with mask'
                color = tuple(colors[0])
            else:
                label = 'person without mask'
                color = tuple(colors[1])

            draw.text((x1 + 4, y1 + 2), label, fill=color)
            draw.rectangle((x1, y1, x2, y2), outline=color)

        drawn_image_path = os.path.join(output_dir, 'output_' + str(step * batch_size + i) + '.jpg')
        print(drawn_image_path)
        array_of_all_images[i].save(drawn_image_path, 'JPEG')
