#include "common.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

int rowsCutoff;
int stesoso = 0;

std::vector<particle_t> local_particles;
std::vector<particle_t> ghost_particles;

double domain_size;
// Put any static global variables here that you will use throughout the simulation.
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
    std::cout << "size " << size << std::endl;
    domain_size = size / num_procs;
    double local_start = rank * domain_size;
    double local_end = local_start + domain_size;
    rowsCutoff = ceil(cutoff / domain_size);

    std::cout << "local start" << local_start << std::endl;
    std::cout << "local end" << local_end << std::endl;

    // Distribute particles across processes
    for (int i = 0; i < num_parts; i += 1) { // Assign each particle to a row based on its position
        int row = parts[i].y / domain_size;
        if (row >= rank && row < rank + 1) {
            local_particles.push_back(parts[i]);
        } else {
            // Add to ghost particles if within cutoff distance based on rowsCutOff
            if (row >= rank - rowsCutoff && row <= rank + rowsCutoff) {
                ghost_particles.push_back(parts[i]);
            }

            // if (parts[i].y >= local_start - cutoff && parts[i].y <= local_end + cutoff) {
            //     ghost_particles.push_back(parts[i]);
            // }
        }
    }

    for (int i = 0; i < num_parts; i++) {
        parts[i].ax = 0;
        parts[i].ay = 0;
    }

    // Print the local particles for each process
    for (int i = 0; i < num_procs; i++) {
        if (i == rank) {
            std::cout << "Process " << rank << " local particles:" << std::endl;
            for (const auto& particle : local_particles) {
                std::cout << "Particle ID: " << particle.id << ", x: " << particle.x
                          << ", y: " << particle.y << std::endl;
            }
            std::cout << "Process " << rank << " ghost particles:" << std::endl;
            for (const auto& particle : ghost_particles) {
                std::cout << "Particle ID: " << particle.id << ", x: " << particle.x
                          << ", y: " << particle.y << std::endl;
            }
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this functions
    stesoso++;
    // output the local particles for each process
    if (stesoso < 5) {
        for (int i = 0; i < num_procs; i++) {
            if (i == rank) {
                std::cout << "Process " << rank << " local particles:" << std::endl;
                for (const auto& particle : local_particles) {
                    std::cout << "Particle ID: " << particle.id << ", x: " << particle.x
                              << ", y: " << particle.y << std::endl;
                }
                std::cout << "Process " << rank << " ghost particles:" << std::endl;
                for (const auto& particle : ghost_particles) {
                    std::cout << "Particle ID: " << particle.id << ", x: " << particle.x
                              << ", y: " << particle.y << std::endl;
                }
                std::cout << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Clear forces
    for (int i = 0; i < local_particles.size(); i++) {
        local_particles[i].ax = 0;
        local_particles[i].ay = 0;
    }

    // Calculate forces for local particles
    for (int i = 0; i < local_particles.size(); i++) {
        for (int j = 0; j < local_particles.size(); j++) {
            apply_force(local_particles[i], local_particles[j]);
        }
        // Additionally calculate forces with relevant ghost particles
        for (int j = 0; j < ghost_particles.size(); j++) {
            apply_force(local_particles[i], ghost_particles[j]);
        }
    }

    // Move particles
    for (int i = 0; i < local_particles.size(); i++) {
        move(local_particles[i], size);
    }

    // std::cout << "Moving" << std::endl;

    // check if particles need to be sent, and make sure to send them to processors where they would
    // be ghost particles
    for (int i = 0; i < local_particles.size(); i++) {
        int row = local_particles[i].y / domain_size;
        if (row < rank || row > rank) {
            // remove from local particles
            local_particles.erase(local_particles.begin() + i);
            for (int i = -rowsCutoff; i <= rowsCutoff; i++) {
                if (row + i >= 0 && row + i < num_procs) {
                    MPI_Request send_request;
                    MPI_Isend(&local_particles[i], 1, PARTICLE, row + i, 0, MPI_COMM_WORLD,
                              &send_request);
                }
            }
        }
    }

    // std::cout << "Sending" << std::endl;
    // Receive particles from other processors and check if they are ghost particles otherwise add
    // them to local particles
    for (int i = 0; i < num_procs; i++) {
        if (i != rank) {
            // std::cout << "Kind of stuck" << std::endl;
            // MPI_Status status;
            // int count;
            // std::cout << "Probe b4" << std::endl;
            // MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
            // std::cout << "Probe" << std::endl;
            // MPI_Get_count(&status, PARTICLE, &count);
            // std::cout << "Count: " << count << std::endl;
            // std::vector<particle_t> received_particles(count);
            // MPI_Request recv_request;
            // std::vector<particle_t> recv_buffer(count);
            particle_t received;
            MPI_Request request;
            MPI_Irecv(&received, sizeof(particle_t), MPI_BYTE, i, 0, MPI_COMM_WORLD, &request);

            // std::cout << "Receiving Function" << std::endl;
            // for (int j = 0; j < count; j++) {
            int row = received.y / domain_size;
            if (row == rank) {
                local_particles.push_back(received);
            } else {
                if (std::find_if(ghost_particles.begin(), ghost_particles.end(),
                                 [&received](const particle_t& p) {
                                     return p.id == received.id;
                                 }) == ghost_particles.end()) {
                    ghost_particles.push_back(received);
                }
            }
            // }
        }
    }

    // std::cout << "Receiving" << std::endl;
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    std::cout << "Gathering" << std::endl;
}