function images = parse(line)
            images = [];
            
            split = regexp(line,'\: ','split');
            
            filename = split{1};
            ranges = split{2};
            
            indexes = regexp(ranges,'\,','split');
            
            for i=1:length(indexes)
                split = regexp(indexes{i},'\-','split');
                first   = str2num(split{1});
                last    = str2num(split{2});
                    
                for j=first:last
                    prefix = regexp(filename,'\_png','split');
                    if j>=100
                        suffix = ['_f' num2str(j) '.png'];
                    elseif j>=10
                        suffix = ['_f0' num2str(j) '.png'];
                    else
                        suffix = ['_f00' num2str(j) '.png'];
                    end
                    img = [prefix{1} suffix];
                    images = [images, string(img)];
                end
                             
            end

        end

