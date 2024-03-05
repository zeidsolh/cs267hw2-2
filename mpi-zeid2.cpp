#include "common.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

struct Cell
{
    std::vector<int> particles; // Indices of particles in each cell
};

// Global variables
std::vector<Cell> grid; // Grid of cells
int grid_row_size;      // Number of cells along one side of the grid
int grid_col_size;      // Number of cells along one side of the grid
int grid_size_sq;       // grid_size * grid_size
double cell_width;
double cell_height;
std::vector<std::pair<int, int>> changes = {
    {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {0, 0}}; // Changes in row and column
std::vector<particle_t> particles;                // Vector of particles for this process
std::vector<particle_t> lower_particles;          // Vector of particles for the lower neighbor
std::vector<particle_t> upper_particles;          // Vector of particles for the upper neighbor

// Apply the force from neighbor to particle
void apply_force(particle_t &particle, particle_t &neighbor)
{
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
void move(particle_t &p, double size)
{
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size)
    {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size)
    {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t *parts, int num_parts, double size, int rank, int num_procs)
{
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here

    double band_size = ((double)size) / ((double)num_procs);
    double y_min = ((double)rank) * band_size;
    double y_max = ((double)(rank + 1)) * band_size;
    // double upper_neighbor_y = size * minimum(rank - 1, 0) / num_procs;
    // grid_size = ceil(band_size / cutoff) + (2 * cutoff); --------- SLOWER VERSION - neighboring
    // rows of cells
    grid_row_size = ceil(band_size / cutoff) * 3;
    grid_col_size = ceil(size / cutoff);
    grid_size_sq = grid_row_size * grid_col_size;
    cell_width = size / grid_col_size;
    cell_height = band_size / grid_row_size;
    grid.resize(grid_size_sq);
    if (rank == 0)
    {
        y_min = 0;
    }
    double lower_neighbor_y = y_min - band_size;
    double upper_neighbor_y = y_max + band_size;
    if (lower_neighbor_y < 0)
    {
        lower_neighbor_y = 0;
    }
    if (upper_neighbor_y > size)
    {
        upper_neighbor_y = size;
    }

    if (band_size <= cutoff)
    {
        std::cerr << "Band size is too small for cutoff" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Iterate over particles, check if y is in our band and add to particles if so
    for (int i = 0; i < num_parts; ++i)
    {
        if (parts[i].y >= y_min && parts[i].y < y_max)
        {
            particles.push_back(parts[i]);
        }
        if (parts[i].y >= lower_neighbor_y && parts[i].y < y_min)
        {
            lower_particles.push_back(parts[i]);
        }
        if (parts[i].y >= y_max && parts[i].y < upper_neighbor_y)
        {
            upper_particles.push_back(parts[i]);
        }
        if (parts[i].y == size && rank == num_procs - 1)
        { // If there is a particle on the edge
            particles.push_back(parts[i]);
        }
        if (parts[i].y == size && rank == num_procs - 2)
        { // If there is a particle on the edge
            upper_particles.push_back(parts[i]);
        }
    }

    // Assert count of all particles across all processes is still the same
    // // std::cout << "0-" << rank << " num particles: " << particles.size() << std::endl;
    // int total_particles = particles.size();
    // int total_lower_particles = lower_particles.size();
    // int total_upper_particles = upper_particles.size();
    // MPI_Allreduce(MPI_IN_PLACE, &total_particles, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // assert(total_particles == num_parts);
}

void simulate_one_step(particle_t *parts, int num_parts, double size, int rank, int num_procs)
{

    // std::cout << "1-" << rank << " num particles: " << particles.size() << std::endl;

    double band_size = ((double)size) / ((double)num_procs);
    double y_min = ((double)rank) * band_size;       // band lower bound
    double y_max = ((double)(rank + 1)) * band_size; // band upper bound

    // Compute forces (just for the particles in our band)
    for (int i = 0; i < grid_size_sq; i += 1)
    {
        grid[i].particles.clear();
    }

    // for every particle in particles and neighbors push to grid
    for (int i = 0; i < particles.size(); i += 1)
    {
        int cell_x = particles[i].x / cell_width;
        int cell_y = particles[i].y / cell_height;
        grid[cell_y * grid_col_size + cell_x].particles.push_back(i);
    }
    for (int i = 0; i < lower_particles.size(); i += 1)
    {
        int cell_x = lower_particles[i].x / cell_width;
        int cell_y = lower_particles[i].y / cell_height;
        // if (cell_y >= (y_min - cutoff) && cell_y < y_min) { --------- SLOWER VERSION -
        // neighboring rows of cells
        grid[cell_y * grid_col_size + cell_x].particles.push_back(i + particles.size());
        // } --------- SLOWER VERSION - neighboring rows of cells
    }
    for (int i = 0; i < upper_particles.size(); i += 1)
    {
        int cell_x = upper_particles[i].x / cell_width;
        int cell_y = upper_particles[i].y / cell_height;
        // if (cell_y >= y_max && cell_y < (y_max + cutoff)) { --------- SLOWER VERSION -
        // neighboring rows of cells
        grid[cell_y * grid_col_size + cell_x].particles.push_back(i + particles.size() +
                                                                  lower_particles.size());
        // } --------- SLOWER VERSION - neighboring rows of cells
    }

    for (int cell_id = 0; cell_id < grid.size(); cell_id += 1)
    {
        int cell_row = cell_id / grid_col_size;
        int cell_col = cell_id % grid_col_size;

        for (int p_id : grid[cell_id].particles)
        {

            if (p_id >= particles.size())
            { // Only compute forces on particles in our band
                continue;
            }

            for (auto change : changes)
            {
                int neighbor_row = cell_row + change.first;
                int neighbor_col = cell_col + change.second;

                if (neighbor_row >= 0 && neighbor_row < grid_row_size && neighbor_col >= 0 &&
                    neighbor_col < grid_col_size)
                {
                    int neighbor_cell_id = neighbor_row * grid_col_size + neighbor_col;

                    for (int neighbor_p_id : grid[neighbor_cell_id].particles)
                    {
                        if ((cell_id != neighbor_cell_id) || (p_id < neighbor_p_id))
                        {
                            if ((neighbor_p_id < particles.size()))
                            {
                                apply_force(particles[p_id], particles[neighbor_p_id]);
                                apply_force(particles[neighbor_p_id], particles[p_id]);
                            }
                            else if ((neighbor_p_id >= particles.size()) &&
                                     neighbor_p_id <
                                         (particles.size() + lower_particles.size()))
                            {
                                apply_force(particles[p_id],
                                            lower_particles[neighbor_p_id - particles.size()]);
                            }
                            else
                            {
                                apply_force(particles[p_id],
                                            upper_particles[neighbor_p_id - particles.size() -
                                                            lower_particles.size()]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Move particles
    for (int i = 0; i < particles.size(); ++i)
    {
        move(particles[i], size);
    }

    // Clear particle accelerations
    for (int i = 0; i < particles.size(); ++i)
    {
        particles[i].ax = particles[i].ay = 0;
    }

    // std::cout << "2-" << rank << " num particles: " << particles.size() << std::endl;

    // Ask neighbors which particles entered our band, update our master copy of particles, then
    // send our master copy to our neighbors
    std::vector<particle_t>
        particles_moving_upwards; // particles moving upwards and our of our band. We will overwrite
                                  // this as the particles we append to our lower neighbor list
    std::vector<particle_t>
        particles_moving_downwards; // particles moving downwards and out of our band. We will
                                    // overwrite this as the particles we append to our upper
                                    // neighbor list

    // If any particles exited our band by going up, append it to the vector
    // If any particles exited our band by going down, append it to the vector
    for (int i = 0; i < particles.size(); ++i)
    {
        if (particles[i].y < y_min)
        {
            particles_moving_downwards.push_back(particles[i]);
            particles.erase(particles.begin() + i); // we need to erase the particle from our list
                                                    // if our band master copy is to stay correct
        }
        if (particles[i].y >= y_max)
        {
            particles_moving_upwards.push_back(particles[i]);
            particles.erase(particles.begin() + i); // we need to erase the particle from our list
                                                    // if our band master copy is to stay correct
        }
    }

    // std::cout << "2.5-" << rank << " num particles upwards: " << particles_moving_upwards.size()
    // << " num particles downwards: " << particles_moving_downwards.size() << std::endl;

    // std::cout << "3-" << rank << " num particles: " << particles.size() << std::endl;

    // Communicate with the appropriate neighbors, make sure to also probe and get count of the
    // number of elements we are sending
    if (rank > 0)
    { // send particles_downwards to the neighbor below us
        MPI_Send(particles_moving_downwards.data(), particles_moving_downwards.size(), PARTICLE,
                 rank - 1, 0, MPI_COMM_WORLD);
    }
    particles_moving_downwards.clear();

    if (rank < num_procs - 1)
    { // receive particles_moving_downwards from the neighbor above us
        MPI_Status status;
        MPI_Probe(rank + 1, 0, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, PARTICLE, &count);
        particles_moving_downwards.resize(count);
        MPI_Recv(particles_moving_downwards.data(), count, PARTICLE, rank + 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    if (rank < num_procs - 1)
    { // send particles_moving_upwards to the neighbor above us
        MPI_Send(particles_moving_upwards.data(), particles_moving_upwards.size(), PARTICLE,
                 rank + 1, 0, MPI_COMM_WORLD);
    }
    particles_moving_upwards.clear();

    if (rank > 0)
    { // receive particles_upwards from the neighbor below us
        MPI_Status status;
        MPI_Probe(rank - 1, 0, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, PARTICLE, &count);
        particles_moving_upwards.clear();
        particles_moving_upwards.resize(count);
        MPI_Recv(particles_moving_upwards.data(), count, PARTICLE, rank - 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    // std::cout << "4-" << rank << " num particles upwards: " << particles_moving_upwards.size() <<
    // " num particles downwards: " << particles_moving_downwards.size() << std::endl;

    // Append the particles to our master copy since these are now the lists of particles that have
    // entered our band
    for (int i = 0; i < particles_moving_downwards.size(); ++i)
    {
        particles.push_back(particles_moving_downwards[i]);
    }
    for (int i = 0; i < particles_moving_upwards.size(); ++i)
    {
        particles.push_back(particles_moving_upwards[i]);
    }

    // std::cout << "6-" << rank << " num particles: " << particles.size() << std::endl;

    // Send our new master copy and receive updated neighbors
    // Send our master copy to the neighbors below
    if (rank > 0)
    {
        MPI_Send(particles.data(), particles.size(), PARTICLE, rank - 1, 0, MPI_COMM_WORLD);
    }
    // Receive the updated neighbors from the neighbors above
    if (rank < num_procs - 1)
    {
        MPI_Status status;
        MPI_Probe(rank + 1, 0, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, PARTICLE, &count);
        upper_particles.resize(count);
        MPI_Recv(upper_particles.data(), count, PARTICLE, rank + 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    // Send our master copy to the neighbors above
    if (rank < num_procs - 1)
    {
        MPI_Send(particles.data(), particles.size(), PARTICLE, rank + 1, 0, MPI_COMM_WORLD);
    }
    // Receive the updated neighbors from the neighbors below
    if (rank > 0)
    {
        MPI_Status status;
        MPI_Probe(rank - 1, 0, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, PARTICLE, &count);
        lower_particles.resize(count);
        MPI_Recv(lower_particles.data(), count, PARTICLE, rank - 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    // std::cout << "7-" << rank << " num particles: " << particles.size() << std::endl;

    // Assert count of all particles across all processes is still the same
    // // std::cout << "7.5-" << rank << " num particles: " << particles.size() << std::endl;
    // int total_particles = particles.size();
    // int total_lower_particles = lower_particles.size();
    // int total_upper_particles = upper_particles.size();
    // MPI_Allreduce(MPI_IN_PLACE, &total_particles, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // assert(total_particles == num_parts);
}

void gather_for_save(particle_t *parts, int num_parts, double size, int rank, int num_procs)
{
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    // Iterate over rank, process 0 will receive and append, processes 1 to num_procs - 1 will send
    // Then append particles in band of rank 0 to the particles list
    // make sure that we re-populate parts with the particles in the correct order according to
    // particle ID

    // std::cout << "8-" << rank << std::endl;

    if (rank == 0)
    {
        for (int i = 0; i < particles.size(); ++i)
        {
            parts[i] = particles[i];
        }
        int parts_idx = particles.size();
        for (int i = 1; i < num_procs; ++i)
        {
            MPI_Status status;
            MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, PARTICLE, &count);
            std::vector<particle_t> particles_from_band_i(count);
            MPI_Recv(particles_from_band_i.data(), count, PARTICLE, i, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            for (int j = 0; j < count; ++j)
            {
                parts[parts_idx] = particles_from_band_i[j];
                parts_idx++;
            }
            assert(parts_idx <= num_parts);
        }
        for (int i = 0; i < particles.size(); ++i)
        {
            parts[i] = particles[i];
        }
        std::sort(parts, parts + num_parts, [](particle_t a, particle_t b)
                  { return a.id < b.id; });
    }
    else
    {
        MPI_Send(particles.data(), particles.size(), PARTICLE, 0, 0, MPI_COMM_WORLD);
    }
}

/*

TODOS:
- Get the faster serial code working for the force computation
- Put up guardrails on if we have too many processes

*/