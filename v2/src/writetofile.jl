open("./outputs/output.txt", "w") do f
    for i in 1:1000
       n1, n2, n3, n4 = rand(1:10,4)
       write(f, "$n1, $n2, $n3, $n4 \n")
       flush(f)
       sleep(0.5)
    end
endd