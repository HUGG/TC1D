CC = gcc
#CFLAGS = -g -Wall -I.
CFLAGS = -O2 -I.
LIBS = -lm
DEPS = ketch.h
OBJ = ketch.c ketch_aft.c
DESTDIR = $(HOME)/.local/bin/

# Compiles source files
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Creates executable locally from object files
ketch_aft: $(OBJ)
	@echo "Building ketch_aft binary in local directory"
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

# Install executable in bin directory
install: ketch_aft
	@echo "Installing ketch_aft binary in $(DESTDIR)"
	@mkdir -p $(DESTDIR) && cp -f ketch_aft $(DESTDIR)/.

# Clean up local object and executable files
clean:
	@echo "Cleaning up object and executable files"
	@rm -rvf *.o
	@rm -rvf ketch_aft
	@rm -rvf ketch_aft.dSYM

# Remove installed executable file
distclean: clean
	@echo "Cleaning up installed executable files"
	@rm -rvf $(DESTDIR)/ketch_aft
