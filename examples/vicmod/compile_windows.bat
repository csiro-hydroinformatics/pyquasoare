del *.obj, *.exe
cl /W4 %~dp0..\..\src\pyquasoare\c_quasoare_utils.c c_quasoare_utils.obj
cl /W4 %~dp0..\..\src\pyquasoare\c_quasoare_core.c c_quasoare_core.obj
cl /W4 example_vicmod.c example_vicmod.obj
link *.obj /OUT:"example_vicmod.exe"
example_vicmod.exe
