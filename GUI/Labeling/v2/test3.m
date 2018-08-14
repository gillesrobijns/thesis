folder = uigetdir;

[name,path] = uigetfile('*.txt');
fullpath= fullfile(path,name);
textfile = fopen(fullpath);
            
data = parse3(textfile);
videopath = fullfile(folder,char(data(1,1)));
video = VideoReader(videopath);
nframes = video.NumberOfFrames;
 
frame = read(video,1);
imshow(frame)