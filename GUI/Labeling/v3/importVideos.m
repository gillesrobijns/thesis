function data = importVideos(textfile,folder,count)
            
            data        = {};
            frames      = {};
            framenbs    = {};
            names       = {};
            k           = 0;
            
            line1 = fgetl(textfile);
            
        
            while ischar(line1) && k<count
                
                
                split       = regexp(line1,'\: ','split');
                filename    = split{1};
                ranges      = split{2};
                
                split       = regexp(filename,'/','split');
                foldersplit = split{1};
                namesplit   = split{2};
                
                
                split       = regexp(namesplit,'\_png','split');
                vidname     = [split{1} '.mp4'];
                fullpath    = [folder '/' vidname];
                
                indexes     = regexp(ranges,'\,','split');
                
                if contains(folder,foldersplit)
                    
                    if exist(fullpath)
                    
                        video       = VideoReader(fullpath);

                        for i=1:length(indexes)
                            split   = regexp(indexes{i},'\-','split');
                            first   = str2num(split{1});
                            last    = str2num(split{2});   
                            for j=first:last
                                frame       = read(video,j);
                                frames      = [frames ; frame];
                                framenbs    = [framenbs ; j];
                                names       = [names ; vidname];
                                k           = k+1;
                            end
                        end  

                        data = [names  framenbs frames];
                    end
                end
                
                line1    = fgetl(textfile);
            end
        end
