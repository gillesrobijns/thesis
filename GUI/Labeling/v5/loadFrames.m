function data = loadFrames(framelist,folder,start,stop)
    
    vidname         = framelist{start,1};
    fullpath        = [folder '/' vidname];
    video           = VideoReader(fullpath);

    for j=start:stop
        
        if strcmp(framelist{j,1},vidname)
            framenb         = framelist{j,2};
            frame           = read(video,framenb);
            framelist{j,3}  = frame;
        else
            vidname         = framelist{j,1};
            fullpath        = [folder '/' vidname];
            video           = VideoReader(fullpath);
            framenb         = framelist{j,2};
            frame           = read(video,framenb);
            framelist{j,3}  = frame;
        end
    end
    
    data = framelist;

end

