    module Metropolis_Ising
        implicit none
        public sweep,many_sweep,many_sweep_measure
    contains    
        subroutine sweep(spins,neighbours,coordination,beta,J,acc,acc_rate,dE_total,dM_total)
            implicit none
            double precision, intent(in) :: beta,J
            double precision, intent(out) :: acc_rate,dE_total
            integer, intent(out) :: dM_total,acc
            integer, intent(in) ::  coordination
            integer,  intent(in)  :: neighbours(:,:)
            integer,  intent(inout) :: spins (:)
            double precision :: dE,u,w2,w4,w6,Delta_E
            integer :: N,sum_NN,h,k,s_i,offset,idx,qabs,Delta_M

            acc = 0
            N = size(spins)
            w2 = exp(-beta * (2.0 * abs(J) * 2.0))
            w4 = exp(-beta * (2.0 * abs(J) * 4.0))
            w6 = exp(-beta * (2.0 * abs(J) * 6.0))
            dE_total = 0.0d0
            dM_total = 0
            call random_number(u)
            offset = int(u * N) + 1
            do h = 0, N-1
                idx = modulo(offset + h - 1, N) + 1
                s_i = spins(idx)        
                sum_NN = 0
                do k = 1, coordination                      
                    sum_NN = sum_NN + spins(neighbours(idx, k))
                end do
                dE = 2.0d0 * J *s_i * sum_NN
                Delta_M = 0
                Delta_E = 0.0d0
                if ( dE <= 0 ) then
                    spins(idx) = -s_i
                    acc = acc + 1
                    Delta_M = -2*s_i
                    Delta_E = dE 
                else
                qabs = abs(s_i * sum_NN)
                call random_number(u)
                select case (qabs)
                case (2)
                    !call random_number(u)
                    if (u <= w2) then 
                        spins(idx) = -s_i 
                        acc = acc + 1
                        Delta_M = -2 * s_i 
                        Delta_E = dE
                    end if
                case (4)
                    !call random_number(u)
                    if (u <= w4) then 
                        spins(idx) = -s_i 
                        acc = acc + 1
                        Delta_M = -2*s_i 
                        Delta_E = dE
                    end if
                case (6)
                    !call random_number(u)
                    if (u <= w6) then 
                        spins(idx) = -s_i 
                        acc = acc + 1
                        Delta_M = -2*s_i
                        Delta_E = dE 
                    end if
                case default
                ! impossible per z=6, ma lasciamo per sicurezza
                !call random_number(u)
                    if (u <= exp(-beta*dE)) then
                        spins(idx) = -s_i 
                        acc = acc + 1
                        Delta_M = -2*s_i 
                        Delta_E = dE
                    end if
                end select
                end if
                dM_total = dM_total + Delta_M
                dE_total = dE_total + Delta_E
            end do
            acc_rate = acc/dble(N)
        end subroutine sweep

        subroutine many_sweep(nequil,nmcs,spins,neighbours,coordination,beta,J, & ! intent in                                           
                              acc,acc_rate, E_array,M_array)               ! intent out
            implicit none
            ! --- Input
            integer,           intent(in)    :: nequil
            integer,           intent(in)    :: nmcs
            integer,           intent(in)    :: coordination
            integer,           intent(in)    :: neighbours(:,:)
            integer,           intent(inout) :: spins(:)
            double precision,  intent(in)    :: beta, J
            !double precision,  intent(in)    :: E0
            !integer,           intent(in)    :: M0
            ! --- Output scalari (ultimo sweep)
            integer,           intent(out)   :: acc
            double precision,  intent(out)   :: acc_rate
            ! --- Output serie (una misura per sweep)
            double precision,  intent(out)   :: E_array(nmcs), M_array(nmcs)

            ! --- Locali
            integer :: i,k,N, dM_single,sum_NN
            double precision :: dE_single
            double precision :: Ecur
            integer :: Mcur,t
            N    = size(spins)
            Ecur = 0.0d0
            do i=1,N
                sum_NN = 0
                do k=1,coordination
                    sum_NN = sum_NN + spins(neighbours(i,k))
                end do
                Ecur = Ecur - 0.5d0 * J * dble(spins(i)*sum_NN)
            end do
            Mcur = 0
            do i=1,N
                Mcur = Mcur + spins(i)
            end do
            Equilibrio:do i = 1, nequil
                call sweep(spins,neighbours,coordination,beta,J,acc,acc_rate, &
                    dE_single,dM_single)
                Ecur = Ecur + dE_single
                Mcur = Mcur + dM_single
            end do Equilibrio

            do i = 1, nmcs
                call sweep(spins,neighbours,coordination,beta,J,acc,acc_rate, &
                           dE_single,dM_single)

                Ecur = Ecur + dE_single
                Mcur = Mcur + dM_single

                ! Salvo densità per spin (coerente con il resto del codice)
                E_array(i) = Ecur / dble(N)
                M_array(i) = dble(Mcur) / dble(N)
            end do
        end subroutine many_sweep
        subroutine many_sweep_measure(nequil,nmcs,measure_step,nsample,spins,neighbours,coordination,beta,J, & ! intent in                                           
                              acc,acc_rate, E_array,M_array,acc_sum_out,nsamples_out,spins_array)               ! intent out
            implicit none
            ! --- Input
            integer,           intent(in)    :: nequil
            integer,           intent(in)    :: nmcs
            integer,           intent(in)    :: measure_step
            integer,           intent(in)    :: nsample
            integer,           intent(inout) :: spins(:)
            integer,           intent(in)    :: neighbours(:,:)
            integer,           intent(in)    :: coordination
            double precision,  intent(in)    :: beta, J
            ! --- Output scalari (ultimo sweep)
            integer,           intent(out)   :: acc
            double precision,  intent(out)   :: acc_rate
            ! --- Output serie (una misura per sweep)
            double precision,  intent(out)   :: E_array(nsample), M_array(nsample)
            integer,           intent(out)   :: acc_sum_out, nsamples_out
            integer,           intent(out)   :: spins_array(size(spins),nsample)

            ! --- Locali
            integer :: i,k, N, dM_single,sum_NN
            double precision :: dE_single
            double precision :: Ecur
            integer :: Mcur
            integer :: t
            N    = size(spins)
            Ecur = 0.0d0
            do i=1,N
                sum_NN = 0
                do k=1,coordination
                    sum_NN = sum_NN + spins(neighbours(i,k))
                end do
                Ecur = Ecur - 0.5d0 * J * dble(spins(i)*sum_NN)
            end do
            Mcur = 0
            do i=1,N
                Mcur = Mcur + spins(i)
            end do
            Equilibrio:do i = 1, nequil
                call sweep(spins,neighbours,coordination,beta,J,acc,acc_rate, &
                    dE_single,dM_single)
                Ecur = Ecur + dE_single
                Mcur = Mcur + dM_single
            end do Equilibrio

            t = 0
            acc_sum_out = 0
            do i = 1, nmcs
            call sweep(spins, neighbours, coordination, beta, J, acc, acc_rate, dE_single, dM_single)
            Ecur = Ecur + dE_single
            Mcur = Mcur + dM_single
            acc_sum_out = acc_sum_out + acc
            if (mod(i-1, measure_step) == 0) then
                t = t + 1
                E_array(t) = Ecur / dble(N)
                M_array(t) = dble(Mcur) / dble(N)
                spins_array(:, t) = spins
            end if
            end do
            nsamples_out = t
        end subroutine many_sweep_measure

    end module Metropolis_Ising