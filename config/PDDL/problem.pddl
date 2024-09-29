(define (problem vineyard_problem)
(:domain vineyard_domain)
(:objects
    rob - robot
    support0 - robot
    l0 - location
    l1 - location
    l2 - location
    l3 - location
    g0 - grape
    g1 - grape
    g2 - grape
    g3 - grape
    b - box
)

(:init
    (robot-at rob l0)
    (support support0)
    (grape-at g0 l0)
    (grape-at g1 l1)
    (grape-at g2 l2)
    (grape-at g3 l3)
    (unchecked l0)
    (unchecked l1)
    (unchecked l2)
    (unchecked l3)
    (free rob)
    (in b rob)
    (adj l0 l1)
    (adj l1 l2)
    (adj l2 l3)
    (full b)
)

(:goal (and (cleared l3)))

)
