folder = uigetdir;
[name,path]     = uigetfile('*.txt');
fullpath        = fullfile(path,name);
textfile        = fopen(fullpath);

frames = importVideos(textfile,folder,100);


