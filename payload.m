%% generate_payload_only_edr2m_iq.m
% Payload-only Bluetooth EDR2M style pi/4-DQPSK IQ
% PRBS9 seed = 0x1FF
% Output CSV: I,Q

clear; clc; close all;

%% Settings
Fs = 20e6;              % sample rate
Rs = 1e6;               % symbol rate
sps = Fs / Rs;          % samples per symbol
numSymbols = 5000;      % 5 ms at 1 Msymbol/s
seed = hex2dec('1FF');  % PRBS9 seed

if mod(sps,1) ~= 0
    error("Fs/Rs must be integer.");
end

%% Generate PRBS9 bits
% PRBS9 polynomial: x^9 + x^5 + 1
% LSB output style
numBits = numSymbols * 2;   % EDR2M: 2 bits per symbol

reg = seed;
bits = zeros(numBits,1);

for k = 1:numBits
    bits(k) = bitand(reg, 1);

    feedback = bitxor(bitget(reg,1), bitget(reg,5));

    reg = bitshift(reg, -1);
    if feedback
        reg = bitor(reg, bitshift(1,8));
    end
    reg = bitand(reg, hex2dec('1FF'));
end

%% Map bits to pi/4-DQPSK phase transitions
% Common mapping:
% 00 -> +pi/4
% 01 -> +3pi/4
% 11 -> -3pi/4
% 10 -> -pi/4

phaseStep = zeros(numSymbols,1);

for k = 1:numSymbols
    b0 = bits(2*k-1);
    b1 = bits(2*k);

    pair = b0*2 + b1;

    switch pair
        case 0  % 00
            phaseStep(k) = pi/4;
        case 1  % 01
            phaseStep(k) = 3*pi/4;
        case 3  % 11
            phaseStep(k) = -3*pi/4;
        case 2  % 10
            phaseStep(k) = -pi/4;
    end
end

%% Generate symbol IQ
phase = cumsum(phaseStep);
symbols = exp(1j * phase);

%% Upsample with rectangular pulse shaping
iq = repelem(symbols, sps);

%% Optional: add frequency offset and noise for testing
addImpairment = false;

if addImpairment
    freqOffset = 50e3;   % 50 kHz offset
    snrDb = 35;          % AWGN SNR

    n = (0:length(iq)-1).';
    iq = iq .* exp(1j*2*pi*freqOffset*n/Fs);

    sigPwr = mean(abs(iq).^2);
    noisePwr = sigPwr / (10^(snrDb/10));
    noise = sqrt(noisePwr/2) * (randn(size(iq)) + 1j*randn(size(iq)));
    iq = iq + noise;
end

%% Save CSV
I = real(iq);
Q = imag(iq);

out = table(I, Q);
writetable(out, "payload_only_edr2m_prbs9_iq_non_ideal.csv");

disp("Generated: payload_only_edr2m_prbs9_iq.csv");
disp("Fs = " + Fs);
disp("Rs = " + Rs);
disp("SPS = " + sps);
disp("Symbols = " + numSymbols);

%% Plot
figure;
plot(real(symbols), imag(symbols), 'o');
grid on; axis equal;
title("Ideal pi/4-DQPSK Symbol Constellation");
xlabel("I"); ylabel("Q");

d = symbols(2:end) .* conj(symbols(1:end-1));
figure;
plot(real(d), imag(d), 'o');
grid on; axis equal;
title("Ideal Differential Constellation");
xlabel("I"); ylabel("Q");