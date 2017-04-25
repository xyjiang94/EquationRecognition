clear all;
close all;
parts = strsplit(mfilename('fullpath'), filesep);
DirPart = mfilename('fullpath');
DirPart = DirPart(1:end-length(parts{end}));
I = imread([DirPart 'equations' filesep 'SKMBT_36317040717260_eq3.png']);
BW = im2bw(I, 0.1);


stats = regionprops(BW);
for index=1:length(stats)
    if stats(index).Area > 20 && stats(index).BoundingBox(3)*stats(index).BoundingBox(4) < 30000
    x = ceil(stats(index).BoundingBox(1))
    y= ceil(stats(index).BoundingBox(2))
    widthX = floor(stats(index).BoundingBox(3)-1)
    widthY = floor(stats(index).BoundingBox(4)-1)
    % TODO
    % if not connected keep left upper
    % if - or . find upper or lower image that might need to merge(need
    % recognize the classification)
    % after getting merged image delete old images
    
    subimage(index) = {BW(y:y+widthY,x:x+widthX,:)}; 
    figure, imshow(subimage{index})
    end
end