#!/usr/bin/awk -f

BEGIN{FS="\t"}

{
    if (!NF) next;
    if ($1 !~ /[0-9_]+/ || $2 !~ /[0-9_]+/){
        print "Bad input: " $0; failed=1; exit 1;
    }
    if ($1 == "_" && $2 == "_"){
        p += 1; r += 1; pd += 1; rd += 1;
    }
    else if ($1 == "_" && $2 != "_") rd += 1;
    else if ($1 != "_" && $2 == "_") pd += 1;
}

END{
    if (failed) exit 1;
    if (!pd) {print "Precision denominator is 0!"; exit 2;}
    if (!rd) {print "Recall denominator is 0!"; exit 2;}
    print "Precision: " p/pd OFS "Recall: " r/rd
}
