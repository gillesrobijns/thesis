folder = uigetdir;
[name,path]     = uigetfile('*.txt');
fullpath        = fullfile(path,name);
textfile        = fopen(fullpath);

metadata = findFrames(textfile,folder);

frames   = loadFrames(metadata,folder,1,100); 



