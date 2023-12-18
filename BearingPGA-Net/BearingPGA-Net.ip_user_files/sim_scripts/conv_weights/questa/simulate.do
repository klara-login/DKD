onbreak {quit -f}
onerror {quit -f}

vsim -t 1ps -lib xil_defaultlib conv_weights_opt

do {wave.do}

view wave
view structure
view signals

do {conv_weights.udo}

run -all

quit -force
