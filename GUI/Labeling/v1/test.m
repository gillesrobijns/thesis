[file,path] = uigetfile('Select Video','*.mp4');
fullpath = fullfile(path,file);
obj = VideoReader(fullpath);

this_frame = readFrame(obj);

imshow(this_frame)

