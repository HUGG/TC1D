CXX = g++
CXXFLAGS = -O2 -I.
#CXXFLAGS = -g -Wall -I.
DEPS = RDAAM.h
OBJ = RDAAM.o RDAAM_He.cpp
DESTDIR = $(HOME)/.local/bin

# Compiles source files
%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

# Creates executable locally from object files
RDAAM_He: $(OBJ)
	@echo "Building RDAAM_He binary in local directory"
	$(CXX) -o $@ $^ $(CXXFLAGS)

# Install executable in bin directory
install: RDAAM_He
	@echo "Installing RDAAM_He binary in $(DESTDIR)"
	@mkdir -p $(DESTDIR) && cp -f RDAAM_He $(DESTDIR)/.

# Clean up local object and executable files
clean:
	@echo "Cleaning up object and executable files"
	@rm -rvf *.o
	@rm -rvf RDAAM_He
	@rm -rvf RDAAM_He.dSYM

# Remove installed executable file
distclean: clean
	@echo "Cleaning up installed executable files"
	@rm -rvf $(DESTDIR)/RDAAM_He
