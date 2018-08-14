function data = importVideos(textfile,folder)

    data  = {};
    line1 = fgetl(textfile);
    
    while ischar(line1)
        
        split       = regexp(line1,'\: ','split');
        filename    = split{1};
        prefix      = regexp(filename,'/','split');
        prefix      = regexp(prefix{2},'\_png','split');
        vidname     = [prefix{1} '.mp4'];
        fullpath    = [folder '/' vidname]
        
        video       = VideoReader(fullpath);
                    
        ranges      = split{2};
        indexes     = regexp(ranges,'\,','split');
        
        for i=1:length(indexes)
            split   = regexp(indexes{i},'\-','split');
            first   = str2num(split{1});
            last    = str2num(split{2});   
            for j=first:last
                frame       = read(video,j);
                data        = [data ; frame];
            end
        end         
        
        
        line1    = fgetl(textfile);
    end
    
end

