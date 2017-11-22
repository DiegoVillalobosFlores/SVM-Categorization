files = dir('monarch_open/*.jpg');

n = length(files);
images = files(1:n,:);

features = 6;

points = zeros(0:n);

for cont=1:n
    name = strcat('monarch_open/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
     a=strongest1.Location;
     a = round(a);
     a = sort(a);
    
    points(cont).x = a(1:features,1);
    points(cont).y = a(1:features,2);
    
%     figure(1);
%     subplot(1,2,1)
%     imshow(I); hold on; plot(strongest1);
end

points = points';


 cont2 = 1;
 for cont=1:n
         for contF=1:features
             coordenadas((cont2),1) = points(cont).x(contF); 
             coordenadas((cont2),2) = points(cont).y(contF); 
             cont2 = cont2 +1;
         end
 end


% cont2 = 1;
% for cont=1:n
%         for contF=1:features
%             coordenadas(cont,contF) = points(cont).x(contF); 
%             coordenadas(cont,contF) = points(cont).y(contF); 
%             cont2 = cont2 +1;
%         end
% end

%coordenadas(1:n,3) = 1;

salida = ones(1,504);

net = feedforwardnet([2 3]);
net = train(net,coordenadas',salida);
A = net(coordenadas');
        
% A = net(b);
% salida_1 = sim(net,b);


%subplot(1,2,2)
%imshow(I2); hold on; plot(strongest2);