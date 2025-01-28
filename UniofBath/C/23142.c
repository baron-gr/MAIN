#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//Functions below

double F(double x0, double x1){
    return ( 100 * pow(x1 - x0*x0, 2.)) + pow(1 - x0, 2.);
}

double max(double N1[3][3]){		//Function to order the points from min to max F(x0,x1) in a matrix
    int n=3,i,j;
    double a,b,c,d,e,f;
    double N[3][3];
    for (i=0;i<n;i++){
        for(j=i+1;j<n;j++){
            if(N1[i][2] > N1[j][2]){
                a =  N1[i][2];
                N1[i][2] = N1[j][2];
                N1[j][2] = a;
                b =  N1[i][2];
                c =  N1[i][1];
                N1[i][1] = N1[j][1];
                N1[j][1] = c;
                d =  N1[i][1];
                e =  N1[i][0];
                N1[i][0] = N1[j][0];
                N1[j][0] = e;
                f =  N1[i][0];
            }
        }
    }
}
	
void replace(double Pa[], double Pb[]){			//Replace Function
    Pa[0] = Pb[0]; Pa[1] = Pb[1]; Pa[2] = Pb[2];
}

double SD(double P1[], double P2[], double P3[], double midP[]){	//Standard deviation Function
    int D = 2; 	//D = no.dimensions
    return sqrt( ( pow((P1[2]-midP[2]),2) + pow((P2[2]-midP[2]),2) + pow((P3[2]-midP[2]),2) ) / D);
}

//For the five functions below, there are only two operations occuring 
//and they could be condensed into just the two functions for simplicity 
//but for ease of identification during main loop
//they have been left as individual named functions.

void midp(double P1[],double P2[],double P3[]){    //Midpoint Function
    int i;
    for(i=0;i<2;i++){
        P3[i] = ((P1[i] + P2[i])/2);
    }
}

void ref(double P1[],double P2[],double P3[]){   //Reflection Function
    int i;
    for(i=0;i<2;i++){
        P3[i] = (2*P1[i] - P2[i]);
    }
}

void expa(double P1[],double P2[],double P3[]){   //Expansion Function
    int i;
    for(i=0;i<2;i++){
        P3[i] = (2*P1[i] - P2[i]);
    }
}

void cont(double P1[],double P2[],double P3[]){   //Contraction Function
    int i;
    for(i=0;i<2;i++){
       	P3[i] = ((P1[i] + P2[i])/2);
    }
}

void fcont(double P1[],double P2[],double P3[]){  //Failed Contraction Function
    int i;
    for(i=0;i<2;i++){
        P3[i] = ((P1[i] + P2[i]))/2;
    }
}


//Main Section


int main(){
	
	//Outputting function to text file
	//Comment out lines 93-99 as you require
	
	FILE* fpointer = fopen("func_output.txt", "w");
    float x0, x1;
    for (x0 = -2; x0 <= 2; x0 += 0.05) {
        x1 = 1.;
        fprintf(fpointer, "%.2f\t%.2f\n", x0, F(x0, x1));
    }
    fclose(fpointer);

    //Lines 102-113 are for input of any starting points, feel free 
    //to comment out 102-113 and uncomment lines 116-118 instead.
    
    float A,B,C,D,E,K;      //F was used in Rosenbrock function so unfortunetly the alphabet will skip a letter.
    double P0[3];
    double P1[3];
    double P2[3];
    printf("\033[1;35m"); 
    printf("Enter starting points for the triangle in the order of:");
    printf("\nP0 = {A,B}, P1 = {C,D}, P2 = {E,K}\n");
    printf("Click enter after each entry\n");
    scanf("\n%f",&A); scanf("\n%f",&B); scanf("\n%f",&C); scanf("\n%f",&D); scanf("\n%f",&E); scanf("\n%f",&K);
	P0[0] = A; P0[1] = B; P1[0] = C; P1[1] = D; P2[0] = E; P2[1] = K;
    P0[2] = F(A,B); P1[2] = F(C,D); P2[2] = F(E,K);
	printf("Your starting values in {x0,x1,F(x0,x1):\nP1={%f,%f,%f}\nP2={%f,%f,%f}\nP3={%f,%f,%f}\n",
	P0[0],P0[1],P0[2],P1[0],P1[1],P1[2],P2[0],P2[1],P2[2]);
	printf("\033[0m");
	
	//Standard starting points
	
// 	double P0[3] = {0,0,F(0,0)}; 
// 	double P1[3] = {2,0,F(2,0)}; 
//	double P2[3] = {0,2,F(0,2)};
 	double PM[3];	//P-bar, the midpoint
 	double Pst[3];	//P*, P_h upon reflection
 	double Pstst[3];	//P**, P_h upon expansion or contraction
	int N = 0;		//Counter for iterations
	
	//Flow-chart
	
	while( SD(P0,P1,P2,PM) > pow(10, -8) && N<1000){        //Criteria

		double A[3][3] = { {P0[0],P0[1],P0[2]}, {P1[0],P1[1],P1[2]}, {P2[0],P2[1],P2[2]} };
		max(A);		//Has ordered points in ascending F(x0,x1)
		
		double PL[3] = {A[0][0],A[0][1],A[0][2]};	//Extracts the, now, ordered arrays based
		double Pi[3] = {A[1][0],A[1][1],A[1][2]};	//on their third value - F(x0,x1)
		double PH[3] = {A[2][0],A[2][1],A[2][2]};
		
		midp(PL,Pi,PM);		//Identifies midpoint of P_low and P_i
		PM[2] = F(PM[0],PM[1]);		//F(x0,x1) for the midpoint coords.
		
		ref(PM, PH, Pst);		//Indentifies reflection point as Pst.
		Pst[2] = F(Pst[0],Pst[1]);
		
		if( Pst[2] < PL[2] ){		//First block in flow-chart
			expa(Pst,PM,Pstst);			//Expansion
			Pstst[2] = F(Pstst[0],Pstst[1]);
				if( Pstst[2] < PL[2] ){
					replace(PH,Pstst);		//Replacement
					replace(P2,Pstst);
					replace(P0,PL);
					replace(P1,Pi);
					P0[2] = F(P0[0],P0[1]);		//Fills in the F(x0,x1) for each point as
					P1[2] = F(P1[0],P1[1]);		//the "replace" function swaps cells 0 and 1,
					P2[2] = F(P2[0],P2[1]);		//after this, cell 2 must be calculated.
				}
				else{
					replace(PH,Pst);		//Replacement
					replace(P2,Pst);
					replace(P0,PL); 
					replace(P1,Pi);
					P0[2] = F(P0[0],P0[1]);
					P1[2] = F(P1[0],P1[1]);
					P2[2] = F(P2[0],P2[1]);
				}
		} 
		else{
			if( Pst[2] > Pi[2]){
				if( Pst[2] > PH[2]){
					cont(PH,PM,Pstst);		//Contraction
					Pstst[2] = F(Pstst[0],Pstst[1]);
					if( Pstst[2] > PH[2]){
						fcont(P0,PL,P0);		//Failed Contraction
						fcont(P1,PL,P1);
						fcont(P2,PL,P2);
						P0[2] = F(P0[0],P0[1]);
						P1[2] = F(P1[0],P1[1]);
						P2[2] = F(P2[0],P2[1]);
					}
					else{
						replace(PH,Pstst);		//Replacement
						replace(P2,Pstst);
						replace(P0,PL); 
						replace(P1,Pi);
						P0[2] = F(P0[0],P0[1]);
						P1[2] = F(P1[0],P1[1]);
						P2[2] = F(P2[0],P2[1]);
					}
				}
				else{
					replace(PH,Pst);		//Replacement followed by Contraction
					cont(PH,PM,Pstst);
					Pstst[2] = F(Pstst[0],Pstst[1]);
						if( Pstst[2] > PH[2] ){
					    	fcont(P0,PL,P0);		//Failed Contraction
						    fcont(P1,PL,P1);
						    fcont(P2,PL,P2);
						    P0[2] = F(P0[0],P0[1]);
						    P1[2] = F(P1[0],P1[1]);
						    P2[2] = F(P2[0],P2[1]);
						}
						else{
							replace(PH,Pstst);		//Replacement
							replace(P2,Pstst);
							replace(P0,PL);
							replace(P1,Pi);
							P0[2] = F(P0[0],P0[1]);
							P1[2] = F(P1[0],P1[1]);
							P2[2] = F(P2[0],P2[1]);
						}
				}
			}
			else{
				replace(PH,Pst);		//Replacement
				replace(P2,Pst);
				replace(P0,PL);
				replace(P1,Pi);
				P0[2] = F(P0[0],P0[1]);
				P1[2] = F(P1[0],P1[1]);
				P2[2] = F(P2[0],P2[1]);
			}
		}
		
		N = N + 1;      //Counter
	}
	
	double X[3][3] = { {P0[0],P0[1],P0[2]}, {P1[0],P1[1],P1[2]}, {P2[0],P2[1],P2[2]} }; //Orders points again
	max(X);		//Has ordered points in ascending F(x0,x1)
	
	printf("\033[1;36m");		//some colour for fun
	printf("\nThe vertices, P(x0,x1,F(x0,x1)), of the smallest triangle are:\nP0={%f,%f,%e}\nP1={%f,%f,%e}\nP2={%f,%f,%e}\n",
	X[0][0],X[0][1],X[0][2],X[1][0],X[1][1],X[1][2],X[2][0],X[2][1],X[2][2]);
	printf("\nWith the closest minimum point PL(x0,x1,F(x0,x1)) = \n{%f,%f,%e}\n", X[0][0],X[0][1],X[0][2]);
	printf("Taking N = %d", N);       //N-1 because printf statement is outside the loop, so it counts an extra iteration
	printf(" iterations.\n");
	printf("\nThese are approximately the confirmed {1,1,0} minimum point.");
	printf("\033[0m");
	
	return(0);
}