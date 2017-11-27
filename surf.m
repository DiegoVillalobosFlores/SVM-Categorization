clear;
close all;

filesOpen = dir('monarch_open/*.jpg');
filesClosed = dir('monarch_closed/*.jpg');

features = 4;

n = length(filesOpen);
m = length(filesClosed);

images = filesOpen(1:n,:);

coordenadas = zeros(features*2,n+m);

for cont=1:n
    name = strcat('monarch_open/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    a=strongest1.Location;
    a = a /10;
    a = round(a);
    %a(1:features,2) = sort(a(1:features,2));
    a = sort(a);
    
    coordenadas(1:features,cont) = a(1:features,1);
    coordenadas(features+1:features*2,cont) = a(1:features,2);
    
    %     figure();
%     imshow(im); hold on; plot(strongest1);
    
end



images = filesClosed(1:m,:);

for cont=1:m
    name = strcat('monarch_closed/', images(cont).name);
    im = imread(name);
    I=rgb2gray(im);

    points1 = detectSURFFeatures(I);
    strongest1 = points1.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    a=strongest1.Location;
    a = a /10;
    a = round(a);
    %a(1:features,2) = sort(a(1:features,2));
    a = sort(a,'descend');

    
    coordenadas(1:features,cont+n) = a(1:features,1);
    coordenadas(features+1:features*2,cont+n) = a(1:features,2);
    
%     figure();
%     imshow(im); hold on; plot(strongest1);
   
end

%  salida(1,1:n) = 1;
%  salida(1,n:n+m) = 0;
%  salida(2,1:n) = 0;
%  salida(2,n:n+m) = 1;

salida(1,1:n) = 1;
salida(1,n:n+m) = 0;


net = patternnet(10);
net = configure(net,coordenadas,salida);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net = train(net,coordenadas,salida);
A = net(coordenadas);
B = round(A);

 figure();
 %plot(1:length(salida),salida,'o',1:n,A(1,1:n),'*',n:n+m,A(2,n:n+m),'*');
 %plot(1:length(salida),salida,'o',1:length(salida),A,'x',1:length(salida),B,'*');
plot(1:length(salida),salida,'o',1:length(salida),A,'*');


files = dir('todas/*.jpg');

k = length(files);
imagen = files(1:k,:);

for cont=1:k
    name = strcat('todas/', imagen(cont).name);
    im = imread(name);
    I=rgb2gray(im);
 
    points3 = detectSURFFeatures(I);
    strongest1 = points3.selectStrongest(features); 
    [featuresSURF, valid_pointsSURF] = extractFeatures(I, strongest1);
    b=strongest1.Location;
    b = b/10;
    b = round(b);
    %a = minmax(a);
    %a = sort(a);
 
     coordenadas2(1:features,cont) = b(1:features,1);
     coordenadas2(features+1:features*2,cont) = b(1:features,2);
end
 
x = net(coordenadas2);
x = round(x);

figure();
plot(1:length(salida),salida,'o',1:length(salida),x,'*');

aciertos = 0;

for cont=1:k
    if(x(1,cont) == salida(1,cont))
        aciertos = aciertos + 1;
    end
end

porcentaje = (aciertos / k) * 100;

% d(1:m,1) = 0;
% d(m+1:m+n,1) = 1;
% 
% promedio = 0;
% 
% for i=1:n+m
%     if d(i,1) == x(i,1)
%         promedio = promedio+1;
%     end
% end
% 
% promedio = round((promedio/(n+m))*100);

%subplot(1,2,2)
%imshow(I2); hold on; plot(strongest2);