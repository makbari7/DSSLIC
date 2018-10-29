% Evaluation Matlab code for "DSSLIC: Deep Semantic Segmentation-based
% Layered Image Compression"
% M. Akbari, J. Liang, and J. Han
% arXiv.org/, June. 2018

% Code tested on Windows 10 (64 bit)

close all;
clear all;

% In order to evaluate the BPG results:
% Download "Binary BPG distribution for Windows (64 bit only)" from https://bellard.org/bpg
% and put all the binary files in a folder named "bpg-win64".
% setenv('PATH', 'bpg-win64');

% For WebP evaluation, you need to install OpenCV from the following link:
% https://github.com/kyamagu/mexopencv
% addpath('.\mexopencv-master\');

global res;
res = 1;

fol = '..\results\Kodak\';
dsImgs=dir(strcat(fol,'*comp*.flif'));
segMaps=dir(strcat(fol,'*map*.flif'));
orgImgs=dir(strcat(fol,'*real*.png'));
recImgs=dir(strcat(fol,'*synthesized*.png'));
% number of files to be processed
numFiles = 1;

% low to high bpp
webp_factors = [1,4,10,25,45,70,82];
j2k_factors =  [62,45,35,28,21,14,11];
jpeg_factors = [9,12,17,23,38,57];
DSSILC_factors = [32,27,23,19,16,14,12,10];
bpg_factors =  [40,37,35,33,30,28,26,24];

[jpeg_mse, jpeg_psnr, jpeg_mssim, jpeg_bpp] = jpegCoding(numFiles, fol, jpeg_factors, orgImgs);
[j2k_mse, j2k_psnr, j2k_mssim, j2k_bpp] = j2kCoding(numFiles, fol, j2k_factors, orgImgs);
[webp_mse, webp_psnr, webp_mssim, webp_bpp] = webpCoding(numFiles, fol, webp_factors, orgImgs);
[bpg_mse, bpg_psnr, bpg_mssim, bpg_bpp] = BpgCoding(numFiles, fol, bpg_factors, orgImgs);
[DSSILC_mse, DSSILC_psnr, DSSILC_mssim, DSSILC_bpp] = DSSILC_Coding(numFiles, fol, DSSILC_factors, orgImgs, recImgs, segMaps,dsImgs);

figure(11)
plot(mean(DSSILC_bpp,1), mean(DSSILC_psnr,1), 'k-.o', 'LineWidth', 1.2, 'MarkerSize', 5);hold on;
plot(mean(bpg_bpp,1), mean(bpg_psnr,1), 'b-.s', 'LineWidth', 1.2, 'MarkerSize', 5);hold on;
plot(mean(webp_bpp,1), mean(webp_psnr,1), 'm-.*', 'LineWidth', 1.2, 'MarkerSize', 5);hold on;
plot(mean(j2k_bpp,1), mean(j2k_psnr,1), 'g-.d', 'LineWidth', 1.2, 'MarkerSize', 5);hold on;
plot(mean(jpeg_bpp,1), mean(jpeg_psnr,1), 'r-.+', 'LineWidth', 1.2, 'MarkerSize', 5);
xlabel('BPP')
ylabel('PSNR (RGB)')
grid on;
legend('DSSLIC (ours)','BPG (4:4:4)','WEBP','JPEG2000','JPEG');

figure(21)
plot(mean(DSSILC_bpp,1), mean(DSSILC_mssim,1), 'k-.o', 'LineWidth', 1.2, 'MarkerSize', 5);hold on;
plot(mean(bpg_bpp,1), mean(bpg_mssim,1), 'b-.s', 'LineWidth', 1.2, 'MarkerSize', 5);hold on;
plot(mean(webp_bpp,1), mean(webp_mssim,1), 'm-.*', 'LineWidth', 1.2, 'MarkerSize', 5);hold on;
plot(mean(j2k_bpp,1), mean(j2k_mssim,1), 'g-.d', 'LineWidth', 1.2, 'MarkerSize', 5);hold on;
plot(mean(jpeg_bpp,1), mean(jpeg_mssim,1), 'r-.+', 'LineWidth', 1.2, 'MarkerSize', 5);
xlabel('BPP')
ylabel('MS-SSIM (RGB)')
grid on;
legend('DSSLIC (ours)','BPG (4:4:4)','WEBP','JPEG2000','JPEG');


function [DSSILC_mse, DSSILC_psnr, DSSILC_mssim, DSSILC_bpp] = DSSILC_Coding(numFiles, fol, DSSILC_factors, orgImgs, recImgs, segMaps,dsImgs)    
    global res;    
    
    DSSILC_mse = zeros(numFiles,numel(DSSILC_factors(1,:)));
    DSSILC_psnr = zeros(numFiles,numel(DSSILC_factors(1,:)));
    DSSILC_mssim = zeros(numFiles,numel(DSSILC_factors(1,:)));
    DSSILC_bpp = zeros(numFiles,numel(DSSILC_factors(1,:)));

    for k=1:numFiles
        dsImage = dsImgs(k).name;
        segMap = segMaps(k).name;
        orgImg = orgImgs(k).name;
        recImg = recImgs(k).name;

        %original image
        org = (imread(strcat(fol,orgImg)));        
        org = double(org);

        map_rec = (imread(strcat(fol,recImg)));        
        map_rec = double(map_rec);        
        
        %get residual, map to [0, 255]
        resi = org - map_rec;
        resi = round((resi + 255) / 2);  % shift then divide by 2 to deal with neg values
        resi = uint8(resi);

        for factor=1:numel(DSSILC_factors)            
            fprintf('--- file: %d, factor: %d\n', k,factor);
            %reconstruct image
            img_rec = map_rec;
            if(res==1)
                Res_Encoding(resi,DSSILC_factors(factor))
                res_rec=Res_Decoding;
                res_rec = double(res_rec);
                %inverse mapping to negative range
                res_rec = 2 * res_rec - 255;

                img_rec = img_rec + res_rec;                
            end

            img_rec(find(img_rec > 255)) = 255;
            img_rec(find(img_rec < 0)) = 0;

            org_ill = uint8(org);
            img_rec_ill = uint8(img_rec);

            DSSILC_mssim(k,factor) = (msssim(img_rec_ill(:,:,1),org_ill(:,:,1))+msssim(img_rec_ill(:,:,2),org_ill(:,:,2))+msssim(img_rec_ill(:,:,3),org_ill(:,:,3)))/3;                        
            DSSILC_mse(k,factor) = immse(org_ill,img_rec_ill);            
            DSSILC_psnr(k,factor) = (psnr(img_rec_ill(:,:,1),org_ill(:,:,1))+psnr(img_rec_ill(:,:,2),org_ill(:,:,2))+psnr(img_rec_ill(:,:,3),org_ill(:,:,3)))/3;

            img_rec = uint8(img_rec);
            imwrite((img_rec),strcat(num2str(k),'DSSILC-rec.png'));            
            tmp_dsImage = dir(strcat(fol,dsImage));
            tmp_segMap = dir(strcat(fol,segMap));
            map_bits = ( tmp_segMap.bytes + tmp_dsImage.bytes) * 8;

            total_bits = map_bits;
            if(res==1)
                tmp1 = dir('DSSILC-res.bpg');
                res_bits = (tmp1.bytes) * 8;
                total_bits = total_bits + res_bits;
            end            
            DSSILC_bpp(k,factor)=total_bits/(size(org,1)*size(org,2)*size(org,3));

            fprintf('DSSILC: R-D information: (%f BPP, %f dB, %f, %f)\n', DSSILC_bpp(k,factor), DSSILC_psnr(k,factor), DSSILC_mse(k,factor), DSSILC_mssim(k,factor));
        end
    end
        
end

function [bpg_mse, bpg_psnr, bpg_mssim, bpg_bpp] = BpgCoding(numFiles, fol, bpg_factors, orgImgs)    
    bpg_mse = zeros(numFiles,numel(bpg_factors));
    bpg_psnr = zeros(numFiles,numel(bpg_factors));
    bpg_mssim = zeros(numFiles,numel(bpg_factors));
    bpg_bpp = zeros(numFiles,numel(bpg_factors));

    for k=1:numFiles
        orgImg = orgImgs(k).name;
        %original image
        org = imread(strcat(fol,orgImg));
        org = double(org);
        for factor=1:numel(bpg_factors)
            fprintf('--- file: %d, factor: %d\n', k,factor);
            
            %%% compression using bpg
            imwrite(uint8(org), 'bpg-rec.png');
            system(['bpgenc -c rgb -q ', num2str(bpg_factors(factor)), ' -o bpg-rec.bpg bpg-rec.png']);
            system(['bpgdec -o ', strcat(num2str(k),'bpg-rec.png'), ' bpg-rec.bpg']);            
            
            org_rec = imread(strcat(num2str(k),'bpg-rec.png')); %read reconstructed bpg

            org_ill = (uint8(org));
            org_rec_ill = (uint8(org_rec));

            bpg_mssim(k,factor) = (msssim(org_rec_ill(:,:,1),org_ill(:,:,1))+msssim(org_rec_ill(:,:,2),org_ill(:,:,2))+msssim(org_rec_ill(:,:,3),org_ill(:,:,3)))/3;
            bpg_mse(k,factor) = immse(org_rec_ill,org_ill);            
            bpg_psnr(k,factor) = (psnr(org_rec_ill(:,:,1),org_ill(:,:,1))+psnr(org_rec_ill(:,:,2),org_ill(:,:,2))+psnr(org_rec_ill(:,:,3),org_ill(:,:,3)))/3;            
                        
            tmp = dir('bpg-rec.bpg');
            total_bits = tmp.bytes * 8;
            bpg_bpp(k,factor) = total_bits/(size(org,1)*size(org,2)*size(org,3));
            %%%
            
            fprintf('bpg: R-D information: (%f BPP, %f dB, %f, %f)\n', bpg_bpp(k,factor), bpg_psnr(k,factor), bpg_mse(k,factor), bpg_mssim(k,factor));
        end
    end        
end

function [jpeg_mse, jpeg_psnr, jpeg_mssim, jpeg_bpp] = jpegCoding(numFiles, fol, jpeg_factors, orgImgs)    
    jpeg_mse = zeros(numFiles,numel(jpeg_factors));
    jpeg_psnr = zeros(numFiles,numel(jpeg_factors));
    jpeg_mssim = zeros(numFiles,numel(jpeg_factors));
    jpeg_bpp = zeros(numFiles,numel(jpeg_factors));

    for k=1:numFiles
        orgImg = orgImgs(k).name;
        %original image
        org = imread(strcat(fol,orgImg));
        org = double(org);
        
        for factor=1:numel(jpeg_factors)
            fprintf('--- file: %d, factor: %d\n', k,factor);
            
            %%% compression using jpeg
            imwrite(uint8(org(:,:,1)), strcat(num2str(k),'jpeg-recR.jpg'), 'Quality', jpeg_factors(factor));  %encode original as jpeg
            imwrite(uint8(org(:,:,2)), strcat(num2str(k),'jpeg-recG.jpg'), 'Quality', jpeg_factors(factor));  %encode original as jpeg
            imwrite(uint8(org(:,:,3)), strcat(num2str(k),'jpeg-recB.jpg'), 'Quality', jpeg_factors(factor));  %encode original as jpeg
            org_recR = imread(strcat(num2str(k),'jpeg-recR.jpg')); %read reconstructed jpeg
            org_recG = imread(strcat(num2str(k),'jpeg-recG.jpg')); %read reconstructed jpeg
            org_recB = imread(strcat(num2str(k),'jpeg-recB.jpg')); %read reconstructed jpeg
            org_rec=cat(3,org_recR,org_recG,org_recB);
            imwrite(uint8(org_rec), strcat(num2str(k),'jpeg-rec.png'));

            org_ill = (uint8(org));
            org_rec_ill = (uint8(org_rec));
                
            jpeg_mssim(k,factor) = (msssim(org_rec_ill(:,:,1),org_ill(:,:,1))+msssim(org_rec_ill(:,:,2),org_ill(:,:,2))+msssim(org_rec_ill(:,:,3),org_ill(:,:,3)))/3; 
            jpeg_mse(k,factor) = immse(org_rec_ill,org_ill);            
            jpeg_psnr(k,factor) = (psnr(org_rec_ill(:,:,1),org_ill(:,:,1))+psnr(org_rec_ill(:,:,2),org_ill(:,:,2))+psnr(org_rec_ill(:,:,3),org_ill(:,:,3)))/3;
            
            tmpR = dir(strcat(num2str(k),'jpeg-recR.jpg'));
            tmpG = dir(strcat(num2str(k),'jpeg-recG.jpg'));
            tmpB = dir(strcat(num2str(k),'jpeg-recB.jpg'));
            total_bits = (tmpR.bytes+tmpG.bytes+tmpB.bytes) * 8;
            jpeg_bpp(k,factor) = total_bits/(size(org,1)*size(org,2)*size(org,3));
            %%%
            
            fprintf('JPEG: R-D information: (%f BPP, %f dB, %f, %f)\n', jpeg_bpp(k,factor), jpeg_psnr(k,factor), jpeg_mse(k,factor), jpeg_mssim(k,factor));
        end
    end
end

function [j2k_mse, j2k_psnr, j2k_mssim, j2k_bpp] = j2kCoding(numFiles, fol, j2k_factors, orgImgs)    
    j2k_mse = zeros(numFiles,numel(j2k_factors));
    j2k_psnr = zeros(numFiles,numel(j2k_factors));
    j2k_mssim = zeros(numFiles,numel(j2k_factors));
    j2k_bpp = zeros(numFiles,numel(j2k_factors));

    for k=1:numFiles
        orgImg = orgImgs(k).name;
        %original image
        org = imread(strcat(fol,orgImg));
        org = double(org);
        for factor=1:numel(j2k_factors)
            fprintf('--- file: %d, factor: %d\n', k,factor);
            
            %%% compression using j2k
            imwrite(uint8(org(:,:,1)), strcat(num2str(k),'j2k-recR.jp2'), 'CompressionRatio', j2k_factors(factor));  %encode original as j2k
            imwrite(uint8(org(:,:,2)), strcat(num2str(k),'j2k-recG.jp2'), 'CompressionRatio', j2k_factors(factor));  %encode original as j2k
            imwrite(uint8(org(:,:,3)), strcat(num2str(k),'j2k-recB.jp2'), 'CompressionRatio', j2k_factors(factor));  %encode original as j2k
            org_recR = imread(strcat(num2str(k),'j2k-recR.jp2')); %read reconstructed j2k
            org_recG = imread(strcat(num2str(k),'j2k-recG.jp2')); %read reconstructed j2k
            org_recB = imread(strcat(num2str(k),'j2k-recB.jp2')); %read reconstructed j2k
            org_rec = cat(3,org_recR,org_recG,org_recB);
            imwrite(uint8(org_rec), strcat(num2str(k),'j2k-rec.png'));

            org_ill = (uint8(org));
            org_rec_ill = (uint8(org_rec));
            
            j2k_mssim(k,factor) = (msssim(org_rec_ill(:,:,1),org_ill(:,:,1))+msssim(org_rec_ill(:,:,2),org_ill(:,:,2))+msssim(org_rec_ill(:,:,3),org_ill(:,:,3)))/3;
            j2k_mse(k,factor) = immse(org_rec_ill,org_ill);            
            j2k_psnr(k,factor) = (psnr(org_rec_ill(:,:,1),org_ill(:,:,1))+psnr(org_rec_ill(:,:,2),org_ill(:,:,2))+psnr(org_rec_ill(:,:,3),org_ill(:,:,3)))/3;            
            
            tmpR = dir(strcat(num2str(k),'j2k-recR.jp2'));
            tmpG = dir(strcat(num2str(k),'j2k-recG.jp2'));
            tmpB = dir(strcat(num2str(k),'j2k-recB.jp2'));
            total_bits = (tmpR.bytes+tmpG.bytes+tmpB.bytes) * 8;
            j2k_bpp(k,factor) = total_bits/(size(org,1)*size(org,2)*size(org,3));
            %%%
            
            fprintf('j2k: R-D information: (%f BPP, %f dB, %f, %f)\n', j2k_bpp(k,factor), j2k_psnr(k,factor), j2k_mse(k,factor), j2k_mssim(k,factor));
        end
    end       
end

function [webp_mse, webp_psnr, webp_mssim, webp_bpp] = webpCoding(numFiles, fol, webp_factors, orgImgs)    
    webp_mse = zeros(numFiles,numel(webp_factors));
    webp_psnr = zeros(numFiles,numel(webp_factors));
    webp_mssim = zeros(numFiles,numel(webp_factors));
    webp_bpp = zeros(numFiles,numel(webp_factors));

    for k=1:numFiles
        orgImg = orgImgs(k).name;
        %original image
        org = imread(strcat(fol,orgImg));
        org = double(org);
        
        for factor=1:numel(webp_factors)
            fprintf('--- file: %d, factor: %d\n', k,factor);
            
            %%% compression using webp                         
            cv.imwrite(strcat(num2str(k),'webp-recR.webp'), uint8(org(:,:,1)), 'WebpQuality', webp_factors(factor));  %encode original as webp
            cv.imwrite(strcat(num2str(k),'webp-recG.webp'), uint8(org(:,:,2)), 'WebpQuality', webp_factors(factor));  %encode original as webp
            cv.imwrite(strcat(num2str(k),'webp-recB.webp'), uint8(org(:,:,3)), 'WebpQuality', webp_factors(factor));  %encode original as webp
            org_recR = cv.imread(strcat(num2str(k),'webp-recR.webp')); %read reconstructed webp
            org_recG = cv.imread(strcat(num2str(k),'webp-recG.webp')); %read reconstructed webp
            org_recB = cv.imread(strcat(num2str(k),'webp-recB.webp')); %read reconstructed webp
            org_rec = cat(3,org_recR(:,:,1),org_recG(:,:,1),org_recB(:,:,1));
            imwrite(uint8(org_rec), strcat(num2str(k),'webp-rec.png'));
            
            org_ill = (uint8(org));
            org_rec_ill = (uint8(org_rec));
            
            webp_mssim(k,factor) = (msssim(org_rec_ill(:,:,1),org_ill(:,:,1))+msssim(org_rec_ill(:,:,2),org_ill(:,:,2))+msssim(org_rec_ill(:,:,3),org_ill(:,:,3)))/3;
            webp_mse(k,factor) = immse(org_rec_ill,org_ill);            
            webp_psnr(k,factor) = (psnr(org_rec_ill(:,:,1),org_ill(:,:,1))+psnr(org_rec_ill(:,:,2),org_ill(:,:,2))+psnr(org_rec_ill(:,:,3),org_ill(:,:,3)))/3;
            
            tmpR = dir(strcat(num2str(k),'webp-recR.webp'));
            tmpG = dir(strcat(num2str(k),'webp-recG.webp'));
            tmpB = dir(strcat(num2str(k),'webp-recB.webp'));
            total_bits = (tmpR.bytes+tmpG.bytes+tmpB.bytes) * 8;
            webp_bpp(k,factor) = total_bits/(size(org,1)*size(org,2)*size(org,3));
            %%%
            
            fprintf('webp: R-D information: (%f BPP, %f dB, %f, %f)\n', webp_bpp(k,factor), webp_psnr(k,factor), webp_mse(k,factor), webp_mssim(k,factor));
        end
    end       
end

function Res_Encoding(res,factor)    
    imwrite(res,'DSSILC-res.png');
    system(['bpgenc -q ', num2str(factor), ' -o DSSILC-res.bpg DSSILC-res.png']);    
    !bpgdec -o DSSILC-res.png DSSILC-res.bpg  
end

function res=Res_Decoding   
    res = imread('DSSILC-res.png');
end

function flifEncoding
    % Download "FLIF Encoder" from https://github.com/FLIF-hub/FLIF and put all the installed binary files in folder "FLIF-master".
    addpath('.\FLIF-master\build\MSVC\')

    fol = '..\results\Kodak\';
    compImgs=dir(strcat(fol,'*comp*.png'));
    segMaps=dir(strcat(fol,'*map*.png'));
    % number of files to be encoded
    numFiles=1;
    for k=1:numFiles
        % comp image coding
        compImg=(imread(strcat(fol,compImgs(k).name)));
        imwrite(compImg,strcat(fol,compImgs(k).name));        
        system(['flif -e ',strcat(fol,compImgs(k).name),' ',strcat(fol,compImgs(k).name),'.flif -E100 --overwrite']);
        % map image coding
        SegMap=imread(strcat(fol,segMaps(k).name));        
        imwrite(SegMap,strcat(fol,segMaps(k).name));
        system(['flif -e ',strcat(fol,segMaps(k).name),' ',strcat(fol,segMaps(k).name),'.flif -E100 --overwrite']);
    end
end