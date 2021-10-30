
.PHONY: clean

MAKEFILES_FOUND = $(shell find . -maxdepth 2 -type f -name Makefile)
SUBDIRS   = $(filter-out ./,$(dir $(MAKEFILES_FOUND)))

clean:
	find . -name "*~" -exec rm {} \;
	for dir in $(SUBDIRS); do \
	make -C $$dir clean; \
	done

all:
	@date
	find . -name "*~" -exec rm {} \;
	for dir in $(SUBDIRS); do \
	echo "=========================";\
	echo "    Making $${dir}";\
	echo "=========================";\
	make -C $$dir all; \
	done
	@date
