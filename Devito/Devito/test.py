
if __name__=="__main__":
    from memory_profiler import profile
    @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def fun():
        a=1
        b=2*10e9
        c=a+b
        del b
        return c
    fun()