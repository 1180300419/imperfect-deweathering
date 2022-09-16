# ROTATION DATA AUGMENTATION CODE
# Code modified from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
import numpy as np
import cv2
import math


def get_translation_matrix_2d(dx, dy):
  """
  Returns a numpy affine transformation matrix for a 2D translation of
  (dx, dy)
  """
  return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

def rotate_image(image, angle):
	"""
	Rotates the given image about it's centre
	"""

	image_size = (image.shape[1], image.shape[0])
	image_center = tuple(np.array(image_size) / 2)

	rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
	trans_mat = np.identity(3)

	w2 = image_size[0] * 0.5
	h2 = image_size[1] * 0.5

	rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

	tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
	tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
	bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
	br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

	x_coords = [pt[0] for pt in [tl, tr, bl, br]]
	x_pos = [x for x in x_coords if x > 0]
	x_neg = [x for x in x_coords if x < 0]

	y_coords = [pt[1] for pt in [tl, tr, bl, br]]
	y_pos = [y for y in y_coords if y > 0]
	y_neg = [y for y in y_coords if y < 0]

	right_bound = max(x_pos)
	left_bound = min(x_neg)
	top_bound = max(y_pos)
	bot_bound = min(y_neg)

	new_w = int(abs(right_bound - left_bound))
	new_h = int(abs(top_bound - bot_bound))
	new_image_size = (new_w, new_h)

	new_midx = new_w * 0.5
	new_midy = new_h * 0.5

	dx = int(new_midx - w2)
	dy = int(new_midy - h2)

	trans_mat = get_translation_matrix_2d(dx, dy)
	affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
	result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

	return result

def rotated_rect_with_max_area(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    # the other two corners are on the mid-line parallel to the longer line
    x = 0.5 * side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return int(wr), int(hr)

# def gen_rotate_image(img, angle):
# 	img = img.transpose(1, 2, 0)
# 	dim = img.shape
# 	h = dim[0]
# 	w = dim[1]

# 	img = rotate_image(img, angle)
# 	dim_bb = img.shape
# 	h_bb = dim_bb[0]
# 	w_bb = dim_bb[1]

# 	w_r, h_r = rotated_rect_with_max_area(w, h, math.radians(angle))

# 	w_0 = (w_bb-w_r) // 2
# 	h_0 = (h_bb-h_r) // 2
# 	img = img[h_0:h_0 + h_r, w_0:w_0 + w_r, :]
# 	img = img.transpose(2, 0, 1)
# 	return img

def gen_rotate_image(img, angle):
	_, h, w = img.shape

	img = img.transpose(1, 2, 0)
	img = rotate_image(img, angle)
	img = img.transpose(2, 0, 1)
	_, r_h, r_w = img.shape

	max_w, max_h = rotated_rect_with_max_area(w, h, math.radians(angle))
	w_half, h_half = (r_w - max_w) // 2, (r_h - max_h) // 2
	img = img[..., h_half : h_half + max_h, w_half : w_half + max_w]
	return img

