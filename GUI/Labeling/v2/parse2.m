function [video,frames] = parse2(line)           
            
    split   = regexp(line,'\: ','split');
    frames  = [];


    filename    = split{1};
    prefix      = regexp(filename,'\_png','split');
    video       = [prefix{1} '.mp4'];
    ranges      = split{2};
            
    indexes = regexp(ranges,'\,','split');
            
            
    for i=1:length(indexes)
        split   = regexp(indexes{i},'\-','split');
        first   = str2num(split{1});
        last    = str2num(split{2});    
        frames  = [frames , first:1:last]
       
    end
    
end

