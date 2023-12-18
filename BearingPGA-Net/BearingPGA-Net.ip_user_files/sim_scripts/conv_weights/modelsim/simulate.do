onbreak {quit -f}
onerror {quit -f}

vsim -voptargs="+acc" -t 1ps -L blk_mem_gen_v8_4_4 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip -lib xil_defaultlib xil_defaultlib.conv_weights xil_defaultlib.glbl

do {wave.do}

view wave
view structure
view signals

do {conv_weights.udo}

run -all

quit -force
