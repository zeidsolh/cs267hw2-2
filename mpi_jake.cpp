#include "common.h"
#include <cmath>
#include "common.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cassert>



struct Cell {
    std::vector<int> particles; // Indices of particles in each cell
};

// Global variables
// std::vector<Cell> grid; // Grid of cells
// int grid_size;          // Number of cells along one side of the grid
// int grid_size_sq;       // grid_size * grid_size
// std::vector<std::pair<int, int>> changes = {
//     {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {0, 0}}; // Changes in row and column
std::vector<particle_t> particles; // Vector of particles for this process
std::vector<particle_t> lower_particles; // Vector of particles for the lower neighbor
std::vector<particle_t> upper_particles; // Vector of particles for the upper neighbor

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    double band_size = ((double) size) / ((double) num_procs);
    double y_min = ((double) rank) * band_size;
    double y_max = ((double) (rank + 1)) * band_size;
    // double upper_neighbor_y = size * minimum(rank - 1, 0) / num_procs;
    if (rank == 0) {
        y_min = 0;
    }
    double lower_neighbor_y = y_min - band_size;
    double upper_neighbor_y = y_max + band_size;
    if (lower_neighbor_y < 0) {
        lower_neighbor_y = 0;
    }
    if (upper_neighbor_y > size) {
        upper_neighbor_y = size;
    }

    if (band_size <= cutoff) {
        std::cerr << "Band size is too small for cutoff" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Iterate over particles, check if y is in our band and add to particles if so
    for (int i = 0; i < num_parts; ++i) {
        if (parts[i].y >= y_min && parts[i].y < y_max) {
            particles.push_back(parts[i]);
        }
        if (parts[i].y >= lower_neighbor_y && parts[i].y < y_min) {
            lower_particles.push_back(parts[i]);
        }
        if (parts[i].y >= y_max && parts[i].y < upper_neighbor_y) {
            upper_particles.push_back(parts[i]);
        }
        if (parts[i].y == size && rank == num_procs - 1) { // If there is a particle on the edge
            particles.push_back(parts[i]);
        }
        if (parts[i].y == size && rank == num_procs - 2) { // If there is a particle on the edge
            upper_particles.push_back(parts[i]);
        }
    }

}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    if (rank == 0) {
        // print out "start"
        std::cout << "1" << std::endl;
    }

    // Compute forces (just for the particles in our band)
        for (int i = 0; i < particles.size(); ++i) {
            particles[i].ax = particles[i].ay = 0;
            for (int j = 0; j < particles.size(); ++j) {
                apply_force(particles[i], particles[j]);
            }
            for (int j = 0; j < lower_particles.size(); ++j) {
                apply_force(particles[i], lower_particles[j]);
            }
            for (int j = 0; j < upper_particles.size(); ++j) {
                apply_force(particles[i], upper_particles[j]);
            }
        }

    // Move particles
        for (int i = 0; i < particles.size(); ++i) {
            move(particles[i], size);
        }

    if (rank == 0) {
        // print out "start"
        std::cout << "2" << std::endl;
    }

    // Send and receive updated neighbors
        if (rank > 0) { // send the particles to the lower neighbor
            MPI_Send(particles.data(), particles.size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD);
        }
        if (rank < num_procs - 1) { // as the lower neighbor, receive the particles, zero should receive and unblock upwards avoiding deadlock
            MPI_Recv(upper_particles.data(), upper_particles.size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < num_procs - 1) { // send the particles to the upper neighbor
            MPI_Send(particles.data(), particles.size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD);
        }
        if (rank > 0) { // as the upper neighbor, receive the particles, num_procs-1 should receive and unblock downwards avoiding deadlock
            MPI_Recv(lower_particles.data(), lower_particles.size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    if (rank == 0) {
        // print out "start"
        std::cout << "3" << std::endl;
    }

    // Rearrange bands for particles that jumped the gap. This covers the case for particles coming into our band (next section is for particles exiting our band)
    // We will delete here any particles from our neighbors that move into our band, appending to our band as we do so
    // Because we will append to 
        // Update which band each particle belongs to by popping and adding particles to the correct band (no communication yet)
        std::vector<particle_t> particles_upwards; // particles moving upwards from lower_neighbor and into our band. We will overwrite this as the particles we append to our upper neighbor list
        std::vector<particle_t> particles_downwards; // particles moving downwards from upper_neighbor and into our band. We will overwrite this as the particles we append to our lower neighbor list
        double band_size = ((double) size) / ((double) num_procs);
        double y_min = ((double) rank) * band_size; // band lower bound
        double y_max = ((double) (rank + 1)) * band_size; // band upper bound

        // If any particles moved into our band from above, append it to the vector
        for (int i = 0; i < upper_particles.size(); ++i) {
            // if the particle is in our band
            if (upper_particles[i].y < y_max) {
                assert (upper_particles[i].y >= y_min); // if this is false, then the particle jumped the gap, which is problematic
                particles_downwards.push_back(upper_particles[i]);
                particles.push_back(upper_particles[i]);
                upper_particles.erase(upper_particles.begin() + i);
            }
        }

        // If any particles moved into our band from below, append it to the vector
        for (int i = 0; i < lower_particles.size(); ++i) {
            // if the particle is in our band
            if (lower_particles[i].y >= y_min) {
                assert (lower_particles[i].y < y_max); // if this is false, then the particle jumped the gap, which is problematic
                particles_upwards.push_back(lower_particles[i]);
                lower_particles.erase(lower_particles.begin() + i);
            }
        }

        // Communicate with the appropriate neighbors, make sure to also probe and get count of the number of elements we are sending
        if (rank > 1) { // send particles_downwards to the neighbor below us
            MPI_Send(particles_downwards.data(), particles_downwards.size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD);
        }

        if (rank < num_procs - 2) { // receive particles_downwards from the neighbor above us
            MPI_Status status;
            MPI_Probe(rank + 1, 0, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, PARTICLE, &count);
            particles_downwards.resize(count);
            MPI_Recv(particles_downwards.data(), count, PARTICLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // this is now the particles we need to append to our list of upper neighbor particles
        }
        
        if (rank < num_procs - 1) { // send particles_upwards to the neighbor above us
            MPI_Send(particles_upwards.data(), particles_upwards.size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD);
        }
        
        if (rank > 0) { // receive particles_upwards from the neighbor below us
            MPI_Status status;
            MPI_Probe(rank - 1, 0, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, PARTICLE, &count);
            particles_upwards.resize(count);
            MPI_Recv(particles_upwards.data(), count, PARTICLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // this is now the particles we need to append to our list of lower neighbor particles
        }

        // Append the particles to the appropriate list
        for (int i = 0; i < particles_upwards.size(); ++i) {
            lower_particles.push_back(particles_upwards[i]);
        }
        for (int i = 0; i < particles_downwards.size(); ++i) {
            upper_particles.push_back(particles_downwards[i]);
        }


    if (rank == 0) {
        // print out "start"
        std::cout << "5" << std::endl;
    }


    // Cull particles that moved out of the main band we have, append them to the appropriate neighbor list
        for (int i = 0; i < particles.size(); ++i) {
            if (particles[i].y < y_min || particles[i].y >= y_max) {
                if (particles[i].y < y_min) {
                    lower_particles.push_back(particles[i]);
                } else {
                    upper_particles.push_back(particles[i]);
                }
                particles.erase(particles.begin() + i);
            }
        }

    
    // Cull particles that moved out of the upper neighbor band or lower neighbor band
        double lower_neighbor_y = y_min - band_size;
        double upper_neighbor_y = y_max + band_size;
        if (lower_neighbor_y < 0) {
            lower_neighbor_y = 0;
        }
        if (upper_neighbor_y > size) {
            upper_neighbor_y = size;
        }
        for (int i = 0; i < lower_particles.size(); ++i) {
            if (lower_particles[i].y < lower_neighbor_y) {
                lower_particles.erase(lower_particles.begin() + i);
            }
        }
        for (int i = 0; i < upper_particles.size(); ++i) {
            if (upper_particles[i].y >= upper_neighbor_y) {
                upper_particles.erase(upper_particles.begin() + i);
            }
        }



}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // Iterate over rank, process 0 will receive and append, processes 1 to num_procs - 1 will send
    // Then append particles in band of rank 0 to the particles list
    // make sure that we re-populate parts with the particles in the correct order according to particle ID

    if (rank == 0) {
        for (int i = 1; i < num_procs; ++i) {
            MPI_Recv(parts, num_parts, PARTICLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < num_parts; ++j) {
                particles.push_back(parts[j]);
            }
        }
        std::sort(particles.begin(), particles.end(), [](particle_t a, particle_t b) { return a.id < b.id; });
        for (int i = 0; i < num_parts; ++i) {
            parts[i] = particles[i];
        }
    } else {
        MPI_Send(particles.data(), particles.size(), PARTICLE, 0, 0, MPI_COMM_WORLD);
    }

}
