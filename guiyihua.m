function image_normalization = guiyihua(image)
image = double(image);
image_minGray = min(min(image));
image_maxGray = max(max(image));
image_distance = image_maxGray-image_minGray;
image_normalization = (image-image_minGray)/image_distance; 
