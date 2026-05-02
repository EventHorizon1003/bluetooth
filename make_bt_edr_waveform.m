%% make_bt_edr_waveform.m
% Generate Bluetooth BR/EDR EDR2M baseband IQ waveform
% Output:
%   1) .mat file containing waveform, Fs, cfg
%   2) .csv file containing I,Q columns
%   3) .bin file containing float32 interleaved I,Q,I,Q...

clear; clc;

%% 0. Check Bluetooth Toolbox function
if exist("bluetoothWaveformGenerator", "file") ~= 2
    error("bluetoothWaveformGenerator not found. You likely do not have MATLAB Bluetooth Toolbox.");
end

disp("Bluetooth waveform generator found. Good. MATLAB is not useless today.");

%% 1. Bluetooth waveform configuration
cfg = bluetoothWaveformConfig;

% EDR2M = pi/4-DQPSK payload, 2 Mbps raw rate
cfg.Mode = "EDR2M";

% Simple EDR packet type
cfg.PacketType = "2-DH1";

% Samples per symbol
% Bluetooth symbol rate = 1 Msymbol/s
% 20 samples/symbol => Fs = 20 MS/s
cfg.SamplesPerSymbol = 20;

%% 2. Payload generation
payloadLengthBytes = getPayloadLength(cfg);
payloadLengthBits = payloadLengthBytes * 8;

payloadBits = randi([0 1], payloadLengthBits, 1);

%% 3. Generate Bluetooth baseband IQ
waveform = bluetoothWaveformGenerator(payloadBits, cfg);

Fs = 1e6 * cfg.SamplesPerSymbol;   % sample rate

%% 4. Normalize waveform
waveform = waveform ./ max(abs(waveform));

%% 5. Add small zero guard before and after packet
guardTime = 20e-6;                 % 20 us guard
guardSamples = round(guardTime * Fs);
guard = complex(zeros(guardSamples, 1));

waveform_guarded = [guard; waveform; guard];

%% 6. Save MATLAB file
save("BT_EDR2M_2DH1_20Msps.mat", ...
     "waveform", "waveform_guarded", "Fs", "cfg", "payloadBits");

%% 7. Save CSV IQ file
iq_csv = [real(waveform_guarded), imag(waveform_guarded)];
writematrix(iq_csv, "BT_EDR2M_2DH1_20Msps_IQ.csv");

%% 8. Save binary float32 interleaved IQ file
iq_interleaved = zeros(2 * length(waveform_guarded), 1, "single");
iq_interleaved(1:2:end) = single(real(waveform_guarded));
iq_interleaved(2:2:end) = single(imag(waveform_guarded));

fid = fopen("BT_EDR2M_2DH1_20Msps_IQ_float32.bin", "w");
fwrite(fid, iq_interleaved, "single");
fclose(fid);

%% 9. Plot quick check
t = (0:length(waveform_guarded)-1).' / Fs;

figure;
plot(t * 1e6, abs(waveform_guarded));
grid on;
xlabel("Time (us)");
ylabel("Magnitude");
title("Bluetooth EDR2M Baseband IQ Magnitude");

figure;
plot(real(waveform_guarded), imag(waveform_guarded), ".");
grid on;
xlabel("I");
ylabel("Q");
title("Bluetooth EDR2M IQ Constellation / Trajectory");

%% 10. Print summary
disp("Generated files:");
disp("1. BT_EDR2M_2DH1_20Msps.mat");
disp("2. BT_EDR2M_2DH1_20Msps_IQ.csv");
disp("3. BT_EDR2M_2DH1_20Msps_IQ_float32.bin");

fprintf("Sample rate Fs = %.2f MS/s\n", Fs/1e6);
fprintf("Samples = %d\n", length(waveform_guarded));
fprintf("Duration = %.2f us\n", length(waveform_guarded)/Fs*1e6);

disp("MXG setting:");
disp("RF Frequency = 2440 MHz");
disp("ARB sample rate = 20 MS/s");
disp("IQ modulation = ON");
disp("RF power = start low, e.g. -40 dBm");