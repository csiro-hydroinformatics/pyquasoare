del *.obj, *.exe
cl /W4 %~dp0\..\src\pyrezeq\c_rezeq_utils.c c_rezeq_utils.obj
cl /W4 %~dp0\..\src\pyrezeq\c_rezeq_quad.c c_rezeq_quad.obj
cl /W4 example.c example.obj
link *.obj /OUT:"example.exe"
example.exe
