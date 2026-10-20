// MIT License; Copyright (c) 2018 Eotvos Lorand University, Budapest.
// Verbatim functions extracted from csviragh/robotsim, commit
// ea54b97e7e2659449e721ebd4a2285beb9b0e9ce. See SOURCE.json and LICENSE.
// Compiled only by tests with a minimal compatibility layer and disk geometry.
double VelDecayLinSqrt(const double x, const double p, const double acc,
        const double v_max, const double r0) {
    //v_max could be named v_diff
    static double vel;
    /* linear v(x) phase */
    vel = (x - r0) * p;
    if (acc <= 0 || p <= 0 || vel <= 0)
        return 0;
    if (vel < acc / p) {
        if (vel >= v_max)
            return v_max;
        return vel;
    }
    /* const acceleration, i.e. sqrt v(x) phase */
    vel = sqrt(2 * acc * (x - r0) - acc * acc / p / p);
    if (vel >= v_max)
        return v_max;
    return vel;
}

void FrictionLinSqrt(double *OutputVelocity, phase_t * Phase,
        const double C_Frict_l, const double V_Frict_l,
        const double Acc_l, const double p_l, const double R_0_l,
        const int WhichAgent, const int Dim_l) {

    NullVect(OutputVelocity, 3);

    int i;
    int n = 0;

    double *AgentsCoordinates;
    double *AgentsVelocity;
    double *NeighboursCoordinates;
    double *NeighboursVelocity;

    AgentsCoordinates = Phase->Coordinates[WhichAgent];
    AgentsVelocity = Phase->Velocities[WhichAgent];

    static double DifferenceVector[3];
    static double DistanceFromNeighbour;
    static double VelDiff;
    static double MaxVelDiff;

    /* Friction-like term */
    for (i = 0; i < Phase->NumberOfAgents; i++) {

        if (i == WhichAgent)
            continue;
        /* Get distance from neighbor */
        NeighboursCoordinates = Phase->Coordinates[i];
        NullVect(DifferenceVector, 3);
        VectDifference(DifferenceVector, NeighboursCoordinates,
                AgentsCoordinates);
        DistanceFromNeighbour = VectAbs(DifferenceVector);
        /* Get velocity difference from neighbor */
        NeighboursVelocity = Phase->Velocities[i];
        VectDifference(DifferenceVector, NeighboursVelocity, AgentsVelocity);
        if (2 == Dim_l) {
            DifferenceVector[2] = 0.0;
        }
        VelDiff = VectAbs(DifferenceVector);
        UnitVect(DifferenceVector, DifferenceVector);
        // calculate max allowed velocity difference at a given distance based
        // on an optimal linsqrt breaking curve and allow for V_Frict slack
        MaxVelDiff =
                MAX(V_Frict_l,
                VelDecayLinSqrt(DistanceFromNeighbour, p_l, Acc_l, VelDiff,
                        R_0_l));
        // if velocity difference is larger than allowed, we compensate it
        if (VelDiff > MaxVelDiff) {
            MultiplicateWithScalar(DifferenceVector, DifferenceVector,
                    C_Frict_l * (VelDiff - MaxVelDiff), Dim_l);
            VectSum(OutputVelocity, OutputVelocity, DifferenceVector);
        }
    }
}

void Shill_Obstacle_LinSqrt(double *OutputVelocity, phase_t * Phase,
        obstacle_t * obstacle, const double V_Shill,
        const double R0_Offset_Shill, const double Acc_Shill,
        const double Slope_Shill, const int WhichAgent) {

    int i;
    double *AgentsPosition = Phase->Coordinates[WhichAgent];
    double *AgentsVelocity = Phase->Velocities[WhichAgent];
    static double ToArena[3];
    static double VelDiff;
    static double DistFromWall; // negative inside obstacle, positive outside
    static double MaxVelDiff;

    // get target point on obstacle wall in ToArena and distance from it in DistFromWall
    // latter will be negative if we are inside obstacle
    DistFromWall =
            DistanceOfNearestPointOfObstacle(ToArena, obstacle, AgentsPosition);
    // inside, shill is going towards arena
    if (DistFromWall < 0) {
        VectDifference(ToArena, ToArena, AgentsPosition);
    }
    // outside, shill is going away from arena
    else {
        VectDifference(ToArena, AgentsPosition, ToArena);
    }
    ToArena[2] = 0;
    UnitVect(ToArena, ToArena);
    MultiplicateWithScalar(ToArena, ToArena, V_Shill, 3);
    VectDifference(ToArena, ToArena, AgentsVelocity);
    ToArena[2] = 0;
    VelDiff = VectAbs(ToArena);
    UnitVect(ToArena, ToArena);
    // calculate max allowed velocity difference at a given distance based
    // on an optimal linsqrt breaking curve
    MaxVelDiff = VelDecayLinSqrt(DistFromWall, Slope_Shill, Acc_Shill,
            VelDiff, R0_Offset_Shill);
    // if velocity difference is larger than allowed, we compensate it
    if (VelDiff > MaxVelDiff) {
        MultiplicateWithScalar(ToArena, ToArena, VelDiff - MaxVelDiff, 2);
        VectSum(OutputVelocity, OutputVelocity, ToArena);
    }
}
