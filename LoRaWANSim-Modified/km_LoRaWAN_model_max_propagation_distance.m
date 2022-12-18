%km_LoRaWAN_model_max_propagation_distance_urban_Hata - calculates maximum
%propagation distance in meters for specified maximum attenuation in dB
%using urban Hata model
%
% Syntax:  [output_distance_meters] = km_LoRaWAN_model_max_propagation_distance_urban_Hata(input_max_loss_dB)
%
% Inputs:
%    input_max_loss_dB - vector of attenuation values in dB
%
% Outputs:
%    output_distance_meters - vector of resulting maximum communication
%    distances in meters
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none

% Author: Konstantin Mikhaylov, Dr. Tech, Wireless Communications
% University of Oulu, Oulu, Finland
% email address: konstantin.mikhaylov(at)oulu.fi
% Website: http://cc.oulu.fi/~kmikhayl/index.html
% March 2020; Last revision: 15-March-2020
function [output_distance_meters] = km_LoRaWAN_model_max_propagation_distance(input_max_loss_dB)
    %SRC: https://en.wikipedia.org/wiki/Hata_model
    output_distance_meters=10.^(input_max_loss_dB/(10.*4.00));
end

