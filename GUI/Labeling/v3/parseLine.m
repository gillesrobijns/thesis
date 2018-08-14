function data = parseLine(line)

    data        = [];
    split       = regexp(line,'\: ','split');
    filename    = split{1};
    prefix      = regexp(filename,'\_png','split');
    vidname     = [prefix{1} '.mp4'];
    ranges      = split{2};
            
    indexes = regexp(ranges,'\,','split');
                
            
     for i=1:length(indexes)
         
        split   = regexp(indexes{i},'\-','split');
        first   = str2num(split{1});
        last    = str2num(split{2}); 
        
        for j=first:last
            data  = [data ; string(vidname),  j];
        end
       
     end
                        
end

