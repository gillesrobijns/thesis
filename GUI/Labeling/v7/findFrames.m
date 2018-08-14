function data = findFrames(textfile,folder)
            
            data        = {};
            frames      = {};
            framenbs    = {};
            names       = {};
            
            line1 = fgetl(textfile);
                    
            while ischar(line1)
                
                
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

                
                    for i=1:length(indexes)
                        split   = regexp(indexes{i},'\-','split');
                        chance  = rand;
                        
                        if chance <0.05
                            first   = 1;
                            last    = 200;
                        else
                            first   = str2num(split{1});
                            last    = str2num(split{2});
                        end
                        for j=first:last
                            framenbs    = [framenbs ; j];
                            names       = [names ; vidname];
                        end
                        
                        
                    end  
                    
                    data = [names  framenbs frames];
                    
                    
                end
                line1    = fgetl(textfile);
            end
        end