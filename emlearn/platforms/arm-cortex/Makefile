BIN=led
OBJS=led.o

# Specify the compiler options specific to the particular microcontroller series
# Ref https://github.com/ARM-software/toolchain-gnu-bare-metal/blob/master/readme.txt
ARM_CORTEX_CFLAGS:=-mthumb -mcpu=cortex-m0plus

# Extra compiler flags, like includes etc
EXTRA_CFLAGS:=""

CC=arm-none-eabi-gcc
OBJCOPY=arm-none-eabi-objcopy
CFLAGS=-Wall -Os ${ARM_CORTEX_CFLAGS} ${EXTRA_CFLAGS} 

${BIN}.hex: ${BIN}.elf
	${OBJCOPY} -O ihex -R .eeprom $< $@

# -lm is needed by models that call libm, such as GradientBoosting
# (expf in sigmoid/softmax) and neural networks. It costs nothing for
# models that do not reference any libm symbol.
${BIN}.elf: ${OBJS}
	${CC} --specs=nosys.specs ${ARM_CORTEX_CFLAGS} -o $@ $^ -lm

clean:
	rm -f ${BIN}.elf ${BIN}.hex ${OBJS}

